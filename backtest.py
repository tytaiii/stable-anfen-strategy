#!/usr/bin/env python3
"""Stable-Anfen Strategy backtest entry point

Usage:
    python backtest.py                                  # default 2026-03-01 -> today
    python backtest.py --start 2025-01-01 --end 2025-12-31
    python backtest.py --start 2026-03-01 --vs-hold     # also compare vs buy & hold
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stable_anfen_strategy as strat

warnings.filterwarnings('ignore')


def calc_stats(eq):
    """Equity series -> performance metrics"""
    ret = eq.iloc[-1] - 1
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    years = days / 365.25
    cagr = (1 + ret) ** (1 / years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    daily = eq.resample('1D').last().pct_change().dropna()
    sharpe = np.sqrt(365) * daily.mean() / daily.std() if daily.std() > 0 else 0.0
    return dict(ret=ret, cagr=cagr, mdd=mdd, sharpe=sharpe)


def slice_eq(d, start, end):
    s = d[(d.index >= start) & (d.index <= end)]['equity']
    return s / s.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-03-01')
    ap.add_argument('--end', default=None, help='defaults to today')
    ap.add_argument('--vs-hold', action='store_true', help='compare against BTC/BNB buy & hold')
    ap.add_argument('--threshold', type=float, default=0.52)
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

    print('>> BTC solo (ML rolling training, ~1-2 min)...', flush=True)
    d_btc, t_btc = strat.run_single_coin('btc', threshold=args.threshold)
    print('>> BNB solo...', flush=True)
    d_bnb, t_bnb = strat.run_single_coin('bnb', threshold=args.threshold)
    print('>> Dual-core rotation 70/30...', flush=True)
    d_combo, t_combo = strat.build_momentum_rotation(
        {'btc': (d_btc, t_btc), 'bnb': (d_bnb, t_bnb)},
        lookback_bars=120, weights_dict={1: 0.7, 2: 0.3})

    # ---- Stats ----
    series = {'Dual-Core Rotation 70/30': slice_eq(d_combo, start, end),
              'BTC Solo': slice_eq(d_btc, start, end),
              'BNB Solo': slice_eq(d_bnb, start, end)}
    trades_map = {'Dual-Core Rotation 70/30': t_combo, 'BTC Solo': t_btc, 'BNB Solo': t_bnb}
    stats = {n: calc_stats(e) for n, e in series.items()}

    print(f"\n===== Backtest window {start.date()} -> {end.date()} =====")
    for name, s in stats.items():
        t = trades_map[name]
        if len(t) > 0 and 'entry_time' in t:
            t = t[(pd.to_datetime(t['entry_time']) >= start) & (pd.to_datetime(t['entry_time']) <= end)]
        wr = (t['pnl'] > 0).mean() if len(t) > 0 else 0.0
        print(f"[{name}] Return {s['ret']*100:+.2f}% | CAGR {s['cagr']*100:+.2f}% | "
              f"MDD {s['mdd']*100:.2f}% | Sharpe {s['sharpe']:.2f} | Trades {len(t)} | Win rate {wr*100:.1f}%")

    # ---- Buy & hold comparison (optional) ----
    holds = {}
    if args.vs_hold:
        for coin in ('btc', 'bnb'):
            df = strat.load_data(coin=coin)
            s = df[(df.index >= start) & (df.index <= end)]
            holds[f'{coin.upper()} Hold'] = s['close'] / s['close'].iloc[0]
        for name, e in holds.items():
            s = calc_stats(e)
            print(f"[{name}] Return {s['ret']*100:+.2f}% | MDD {s['mdd']*100:.2f}% | Sharpe {s['sharpe']:.2f}")

    # ---- Chart ----
    SURFACE = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#898781'
    GRID = '#e1e0d9'; AXIS = '#c3c2b7'
    C_ROT, C_BTC, C_BNB = '#2a78d6', '#eb6834', '#1baf7a'

    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'text.color': INK, 'grid.color': GRID,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 1.2], sharex=True)
    fig.subplots_adjust(hspace=0.06, top=0.88, bottom=0.10, left=0.07, right=0.93)

    # With --vs-hold the comparison lines are buy & hold; otherwise the solo strategies
    if args.vs_hold:
        plot_lines = [('Dual-Core Rotation 70/30', series['Dual-Core Rotation 70/30'], C_ROT, 3.0),
                      ('BTC Hold', holds['BTC Hold'], C_BTC, 2.0),
                      ('BNB Hold', holds['BNB Hold'], C_BNB, 2.0)]
    else:
        plot_lines = [('Dual-Core Rotation 70/30', series['Dual-Core Rotation 70/30'], C_ROT, 3.0),
                      ('BTC Solo', series['BTC Solo'], C_BTC, 2.0),
                      ('BNB Solo', series['BNB Solo'], C_BNB, 2.0)]

    for name, eq, color, lw in plot_lines:
        ax1.plot(eq.index, eq.values, color=color, linewidth=lw, solid_capstyle='round', zorder=3)

    x_last = series['Dual-Core Rotation 70/30'].index[-1]
    ax1.text(x_last, series['Dual-Core Rotation 70/30'].values[-1],
             f'   Rotation {stats["Dual-Core Rotation 70/30"]["ret"]*100:+.1f}%',
             va='center', ha='left', fontsize=11, color=INK, fontweight='bold')
    for name, eq, _, _ in plot_lines[1:]:
        value = stats[name]['ret'] if name in stats else holds[name].iloc[-1] - 1
        ax1.text(x_last, eq.values[-1], f'  {name} {value*100:+.1f}%',
                 va='center', ha='left', fontsize=10, color=INK2)

    ax1.axhline(1.0, color=AXIS, linewidth=1, linestyle='--', zorder=2)
    ax1.set_ylabel('Equity (start = 1.0)', fontsize=11)
    ax1.set_title(f'Stable-Anfen Strategy · {start.date()} to {end.date()} (1x no leverage · fees included)',
                  fontsize=15, fontweight='bold', color=INK, loc='left', pad=10)
    ax1.grid(True, axis='y', alpha=0.6, zorder=1)
    ax1.margins(x=0.06, y=0.12)

    dd = series['Dual-Core Rotation 70/30'] / series['Dual-Core Rotation 70/30'].cummax() - 1
    ax2.fill_between(dd.index, dd.values * 100, 0, color=C_ROT, alpha=0.18, zorder=2)
    ax2.plot(dd.index, dd.values * 100, color=C_ROT, linewidth=2, solid_capstyle='round', zorder=3)
    ax2.axhline(0, color=AXIS, linewidth=1, zorder=1)
    ax2.set_ylabel('Rotation drawdown %', fontsize=11)
    ax2.grid(True, axis='y', alpha=0.6)
    ax2.margins(x=0.06)

    out = f'backtest_{start.date()}_{end.date()}.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print(f'\n[OK] Chart saved: {out}')


if __name__ == '__main__':
    main()
