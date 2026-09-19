"""Tests for data loading and feature engineering."""
import numpy as np
import pandas as pd

import stable_anfen_strategy as strat


def test_load_data_bundled():
    df = strat.load_data(coin='btc')
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert {'open', 'high', 'low', 'close', 'dvol'}.issubset(df.columns)
    assert not df['dvol'].isna().any()
    assert (df['close'] > 0).all()


def test_load_data_coins_aligned():
    df_btc = strat.load_data(coin='btc')
    df_bnb = strat.load_data(coin='bnb')
    assert len(df_btc) == len(df_bnb)
    assert (df_btc.index == df_bnb.index).all()


def test_feature_engineering_columns(make_df):
    df = strat.feature_engineering(make_df())
    for col in ['r', 'zscore', 'z_dvol_90', 'atr14', 'vol', 'dz', 'dz_lag1',
                'trigger', 'z']:
        assert col in df.columns
    for h in strat.HORIZONS_BARS:
        assert f'p_{h}' not in df.columns  # added later by ML training
        assert f'y_{h}' in df.columns


def test_feature_engineering_returns_spec(make_df):
    df = strat.feature_engineering(make_df(n_bars=300))
    # r = log(close/close.shift(1))
    expected_r = np.log(df['close'] / df['close'].shift(1))
    pd.testing.assert_series_equal(df['r'], expected_r, check_names=False)
    # trigger is exactly the spec: dz crosses up through zero
    expected_trigger = ((df['dz_lag1'] < 0) & (df['dz'] > 0)).astype(int)
    pd.testing.assert_series_equal(df['trigger'], expected_trigger, check_names=False)


def test_feature_engineering_labels(make_df):
    df = strat.feature_engineering(make_df(n_bars=300))
    for h in strat.HORIZONS_BARS:
        fwd_ret = df['close'].shift(-h) / df['open'].shift(-1) - 1
        expected = (fwd_ret > 0).astype(int)
        expected.iloc[-h:] = np.nan
        pd.testing.assert_series_equal(df[f'y_{h}'], expected, check_names=False)


def test_trigger_fires_on_dvol_step_down(make_df):
    """A sharp DVOL step-down makes the z-score jump down (dz<0) and then
    relax back up (dz>0) - the strategy's 'volatility bottoming' trigger."""
    df = make_df(n_bars=160)
    dvol = np.full(160, 50.0)
    dvol[80:110] = 42.0
    dvol[110:135] = 36.0  # second step down
    dvol[135:] = 48.0     # rebound
    df['dvol'] = dvol
    df = strat.feature_engineering(df)
    assert df['trigger'].sum() >= 1
    # first trigger lands right after the second step-down (bar ~111)
    first_trigger = df['trigger'].idxmax()
    assert df.index[105] <= first_trigger < df.index[125]


def test_trigger_silent_when_dvol_flat(make_df):
    """Flat DVOL has zero z-score std -> zscore NaN -> no triggers."""
    df = strat.feature_engineering(make_df(n_bars=120))
    assert df['trigger'].sum() == 0
