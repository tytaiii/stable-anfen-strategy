"""Tests for the main-gun trading loop, stats, and dual-gun blending."""
import numpy as np
import pandas as pd

import stable_anfen_strategy as strat

COST = 0.0007


def prepared_df(make_df, n_bars=300):
    """Feature-engineered df with a wide-open safety gate."""
    df = strat.feature_engineering(make_df(n_bars=n_bars))
    df['safety_score'] = 0.8
    df['p_6'] = df['p_18'] = df['p_30'] = 0.8
    df['trigger'] = 0
    return df


def test_gate_closed_never_trades(make_df):
    df = strat.feature_engineering(make_df(n_bars=300))
    df['safety_score'] = 0.3  # below every threshold
    df['trigger'] = 0
    df.loc[df.index[100], 'trigger'] = 1  # would be an entry if the gate were open
    df_eval, trades, metrics = strat.run_strategy(df, mode='1x', threshold=0.52)
    assert len(trades) == 0
    assert (df_eval['equity'] == 1.0).all()
    assert (df_eval['pos'] == 0).all()
    assert metrics['rej_S'] == 1  # the blocked trigger was counted


def test_baseline_entry_exit_roundtrip(make_df):
    df = prepared_df(make_df)
    df.loc[df.index[100], 'trigger'] = 1
    df.loc[df.index[120], 'r'] = -0.01  # make the exit bar's return negative
    df.loc[df.index[120], 'trigger'] = 1
    df_eval, trades, metrics = strat.run_strategy(df, mode='1x', threshold=0.52)

    assert len(trades) == 1
    t = trades.iloc[0]
    entry_price = df['open'].iloc[101] * (1 + COST)
    exit_price = df['open'].iloc[121] * (1 - COST)
    assert np.isclose(t['entry_price'], entry_price)
    assert np.isclose(t['exit_price'], exit_price)
    assert t['entry_time'] == df.index[101]
    assert t['exit_time'] == df.index[121]
    assert t['lev'] == 1.0
    expected_pnl = exit_price / entry_price - 1
    assert np.isclose(t['pnl'], expected_pnl)
    assert np.isclose(df_eval['equity'].iloc[-1], 1 + expected_pnl)


def test_scalar_and_array_threshold_equivalent(make_df):
    df = prepared_df(make_df)
    df.loc[df.index[100], 'trigger'] = 1
    scalar = strat.run_strategy(df, mode='1x', threshold=0.52)[0]['equity']
    array = strat.run_strategy(df, mode='1x', threshold=np.full(len(df), 0.52))[0]['equity']
    np.testing.assert_allclose(scalar.values, array.values)


def test_queued_delay_waits_for_short_score(make_df):
    df = prepared_df(make_df)
    df['p_6'] = df['p_18'] = df['p_30'] = 0.45  # below T_short
    df.loc[df.index[100], 'trigger'] = 1
    # short score recovers at bar 105, within max_wait=12
    df.loc[df.index[105]:, ['p_6', 'p_18', 'p_30']] = 0.6
    df_eval, trades, metrics = strat.run_strategy(
        df, mode='1x', threshold=0.52, delay_method='queued',
        T_short=0.50, max_wait=12)
    assert metrics['rej_delay'] == 1
    assert len(trades) == 1
    assert trades.iloc[0]['entry_time'] == df.index[106]


def test_force_close_on_last_bar(make_df):
    df = prepared_df(make_df)
    df.loc[df.index[100], 'trigger'] = 1
    df_eval, trades, metrics = strat.run_strategy(df, mode='1x', threshold=0.52)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t['exit_time'] == df.index[-1]
    assert np.isclose(t['exit_price'], df['close'].iloc[-1] * (1 - COST))


def test_vol_limiter_rejects_high_vol_entry(make_df):
    df = make_df(n_bars=300, spread=0.05)  # ATR/close ~ 10% > 3% cap
    df = strat.feature_engineering(df)
    df['safety_score'] = 0.8
    df['trigger'] = 0
    df.loc[df.index[100], 'trigger'] = 1
    df_eval, trades, metrics = strat.run_strategy(df, mode='lev_vol', threshold=0.52)
    assert len(trades) == 0
    assert metrics['rej_vol'] >= 1


def test_calc_stats_complex():
    idx = pd.date_range('2023-01-01', periods=4, freq='4h')
    df_eval = pd.DataFrame({'equity': [1.0, 1.1, 1.05, 1.2]}, index=idx)
    trades = pd.DataFrame({'pnl': [0.10, -0.045454545, 0.142857143],
                           'lev': [1.0, 1.0, 1.0]})
    s = strat.calc_stats_complex(df_eval, trades, {'rej_S': 0, 'rej_vol': 0, 'rej_cb': 0})
    assert np.isclose(s['ret'], 0.2)
    assert np.isclose(s['mdd'], 1.05 / 1.1 - 1)
    assert s['trades'] == 3
    assert np.isclose(s['win_rate'], 2 / 3)
    gross_profit = 0.10 + 0.142857143
    gross_loss = abs(-0.045454545)
    assert np.isclose(s['pf'], gross_profit / gross_loss)


def test_calc_stats_empty_trades():
    idx = pd.date_range('2023-01-01', periods=4, freq='4h')
    df_eval = pd.DataFrame({'equity': [1.0, 1.0, 1.0, 1.0]}, index=idx)
    empty = pd.DataFrame(columns=['pnl', 'lev'])
    s = strat.calc_stats_complex(df_eval, empty, {'rej_S': 0, 'rej_vol': 0, 'rej_cb': 0})
    assert s['trades'] == 0
    assert s['win_rate'] == 0
    assert s['pf'] == 0
    assert s['ret'] == 0


def test_dual_gun_blend_bounds(make_df):
    df = strat.feature_engineering(make_df(n_bars=300))
    df['safety_score'] = 0.8
    d_full, t_full, _ = strat.run_strategy(df, mode='1x', threshold=0.52)
    d_dual, t_dual, w_scout = strat.run_dual_gun_blend(df, d_full, t_full)
    assert len(d_dual) == len(df)
    assert np.isfinite(d_dual['equity']).all()
    assert d_dual['equity'].iloc[0] == 1.0
    assert ((w_scout >= 0) & (w_scout <= 0.4)).all()
