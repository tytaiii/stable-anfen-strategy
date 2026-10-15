# (💰 v 💰) Stable-Anfen Strategy

**Manage drawdowns. Follow trends.**

An unleveraged BTC/BNB strategy focused on drawdown control. It combines machine learning signals, trend following, and 70/30 momentum rotation on 4-hour bars.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) [![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE) ![CI](https://github.com/tytaiii/stable-anfen-strategy/actions/workflows/ci.yml/badge.svg)

## How It Works

- **Trade selectively.** The main strategy filters entries with an ML score and exits when that score falls below its threshold. An independent trend module captures breakouts.
- **Favor the stronger strategy.** Compare the BTC and BNB strategies' equity performance over the past 20 days; allocate 70% to the stronger side and 30% to the other.
- **Stay unleveraged.** Both trading modules use 1x exposure. Total portfolio weights never exceed 100% in the default configuration.

## Out-of-Sample Live Validation

**2026-03-01 to 2026-09-18 — forward out-of-sample live validation (author-reported).**

Strategy rules were fixed before this period began. Scheduled updates used only historical information available at the time, with each subsequent decision evaluated on unseen market data.

The table and chart are a **backtest reproduction of the same period**, using the bundled data, rather than audited live-account returns.

| Metric | Stable-Anfen | BTC Buy & Hold | BNB Buy & Hold |
| :--- | ---: | ---: | ---: |
| Total return | **+14.98%** | +19.64% | +20.59% |
| Maximum drawdown | **-6.97%** | -29.25% | -26.67% |
| Sharpe ratio | **1.81** | 1.13 | 1.14 |

In this reproduction, portfolio drawdown was about one quarter of either buy-and-hold benchmark.

![Stable-Anfen portfolio versus BTC and BNB buy and hold](./docs/backtest_2026_sample.png)

Includes fees of 0.07% per side. Excludes funding, additional slippage, and portfolio rebalancing costs. Past performance does not guarantee future results. [Backtest methodology](./docs/strategy.md)

## Quick Start

Requires Python 3.10+. Price, DVOL, and threshold data are included. No API key is needed.

```bash
git clone https://github.com/tytaiii/stable-anfen-strategy.git
cd stable-anfen-strategy
pip install -r requirements.txt

# Reproduce the validation period in backtest and compare against buy and hold
python backtest.py --start 2026-03-01 --end 2026-09-18T16:00 --vs-hold
```

Outputs an equity chart and statistics, including return, maximum drawdown, and Sharpe ratio. Change `--start` and `--end` to explore other periods.

```bash
# Optional: refresh prices and DVOL; precomputed WFA thresholds are not updated
python fetch_data.py
```

## Code & Documentation

| File | Purpose |
| :--- | :--- |
| [stable_anfen_strategy.py](./stable_anfen_strategy.py) | ML scoring, trade signals, trend following, and rotation |
| [backtest.py](./backtest.py) | Backtesting, statistics, and charts |
| [fetch_data.py](./fetch_data.py) | Binance price and Deribit DVOL updates |
| [Strategy guide](./docs/strategy.md) | Parameters, data, and backtest assumptions |
| [Contributing](./CONTRIBUTING.md) | Development setup and tests |

[MIT License](./LICENSE). For quantitative research; not investment advice.
