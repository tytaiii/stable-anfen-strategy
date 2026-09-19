"""Shared fixtures for the test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_df():
    """Factory for synthetic 4h OHLCV+DVOL frames.

    Defaults: gap-free rising closes (+0.1%/bar), high/low at +/-0.2%,
    constant DVOL of 50.
    """
    def _make(n_bars=300, close_pct=0.001, dvol=50.0, spread=0.002):
        idx = pd.date_range('2023-01-01', periods=n_bars, freq='4h')
        close = 100.0 * (1 + close_pct) ** np.arange(n_bars)
        open_ = np.r_[close[0], close[:-1]]
        df = pd.DataFrame({
            'open': open_,
            'high': close * (1 + spread),
            'low': close * (1 - spread),
            'close': close,
            'volume': 1000.0,
        }, index=idx)
        df['dvol'] = float(dvol)
        return df
    return _make
