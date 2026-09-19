# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-19

### Added

- Initial open-source release of the Stable-Anfen Strategy
- Core strategy module (`stable_anfen_strategy.py`): ML safety-score gating
  (6-horizon Logistic Regression, expanding window with time decay and
  critical-event anchoring), dual-gun equity blending, and 70/30 momentum
  rotation over BTC/BNB at constant 1x leverage
- `backtest.py` entry point with window selection and buy & hold comparison
- `fetch_data.py` incremental data fetcher for Binance klines and
  Deribit DVOL (with 12h fallback for older history)
- Bundled sample data (2021-01 → 2026-09, 4h) and sample backtest chart
- Unit test suite (feature engineering, trading loop, scout gun, rotation)
- CI workflow running tests on push and pull requests
