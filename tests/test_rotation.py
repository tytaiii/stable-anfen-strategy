"""Tests for the scout gun and the 70/30 momentum rotation engine."""
import numpy as np
import pandas as pd

import stable_anfen_strategy as strat

COST = 0.0007
EMPTY_TRADES = pd.DataFrame(columns=['entry_time', 'pnl', 'lev'])


def make_coin_eq(idx, bar_pct):
    """Synthetic coin equity with a constant per-bar return."""
    return pd.DataFrame({'equity': np.cumprod(np.full(len(idx), 1 + bar_pct))},
                        index=idx)


def test_rotation_weights_70_30():
    idx = pd.date_range('2023-01-01', periods=400, freq='4h')
    d_btc = make_coin_eq(idx, 0.002)   # winner: +0.2%/bar
    d_bnb = make_coin_eq(idx, -0.002)  # loser: -0.2%/bar
    d_combo, t_combo = strat.build_momentum_rotation(
        {'btc': (d_btc, EMPTY_TRADES.copy()), 'bnb': (d_bnb, EMPTY_TRADES.copy())},
        lookback_bars=120, weights_dict={1: 0.7, 2: 0.3})
    eq = d_combo['equity']
    # no momentum before the 120-bar window -> weights are zero, equity flat
    assert (eq.iloc[:121] == 1.0).all()
    # per-bar: 0.7*0.002 + 0.3*(-0.002) = 0.0008
    expected = 1.0008 ** (len(idx) - 121)
    assert np.isclose(eq.iloc[-1], expected)


def test_rotation_tie_uses_stable_first_rank():
    idx = pd.date_range('2023-01-01', periods=400, freq='4h')
    d_btc = make_coin_eq(idx, 0.002)
    d_bnb = make_coin_eq(idx, 0.002)
    d_combo, _ = strat.build_momentum_rotation(
        {'btc': (d_btc, EMPTY_TRADES.copy()), 'bnb': (d_bnb, EMPTY_TRADES.copy())},
        lookback_bars=120, weights_dict={1: 0.7, 2: 0.3})
    eq = d_combo['equity']
    # tied momentum -> rank 1 goes to the first coin (btc), weights sum to 1
    expected = 1.002 ** (len(idx) - 121)
    assert np.isclose(eq.iloc[-1], expected)


def test_rotation_trade_passthrough_weights():
    idx = pd.date_range('2023-01-01', periods=400, freq='4h')
    d_btc = make_coin_eq(idx, 0.002)
    d_bnb = make_coin_eq(idx, -0.002)
    t_btc = pd.DataFrame({
        'entry_time': [idx[130], idx[50]],  # inside vs before momentum window
        'pnl': [0.05, 0.03], 'lev': [1.0, 1.0],
    })
    d_combo, t_combo = strat.build_momentum_rotation(
        {'btc': (d_btc, t_btc), 'bnb': (d_bnb, EMPTY_TRADES.copy())},
        lookback_bars=120, weights_dict={1: 0.7, 2: 0.3})
    # only the in-window trade survives, weighted at 70%
    assert len(t_combo) == 1
    assert np.isclose(t_combo.iloc[0]['pnl'], 0.05 * 0.7)
    assert np.isclose(t_combo.iloc[0]['port_weight'], 0.7)


def test_scout_gun_trend_roundtrip(make_df):
    """Scout: HH20 breakout entry, 10-bar low exit, constant 1x."""
    df = make_df(n_bars=250, close_pct=0.01)  # +1%/bar -> breaks HH20 every bar
    # crash at bar 160, then flat forever (no re-entry)
    crash = df['close'].iloc[160 - 1] * 0.9
    df.loc[df.index[160]:, 'close'] = crash
    df.loc[df.index[160]:, 'open'] = crash
    df.loc[df.index[160]:, 'high'] = crash * 1.002
    df.loc[df.index[160]:, 'low'] = crash * 0.998

    d_scout, t_scout, _ = strat.scout_strategy_greedy_dvol(df, cost=COST)
    assert len(t_scout) == 1
    t = t_scout.iloc[0]
    assert t['lev'] == 1.0
    assert t['entry_time'] == df.index[121]
    assert np.isclose(t['entry_price'], df['open'].iloc[121] * (1 + COST))
    assert t['exit_time'] == df.index[161]
    assert np.isclose(t['exit_price'], df['open'].iloc[161] * (1 - COST))
