# Strategy & Backtest Methodology

[Back to README](../README.md)

Stable-Anfen runs a main strategy and an independent trend module for each of BTC and BNB, then allocates portfolio weights using the momentum of the two strategies' equity curves. In the default backtest, both trading modules use 1x exposure and are either long or flat.

## 1. Data & Features

- **Prices:** Binance BTCUSDT and BNBUSDT spot candles at 4-hour intervals.
- **Volatility:** Deribit's BTC DVOL implied volatility index, shared by both assets.
- **Model features:** The 60-bar rolling Z-score of DVOL, `z`, and its first difference, `dz`.
- **Turning point:** A crossing of `dz` from negative to positive produces a `trigger`.

## 2. Machine Learning Score

Six logistic regression models estimate the probability of a price increase over 1, 3, 5, 7, 14, and 30 days. The median of these probabilities becomes `safety_score`, a trading signal rather than a guarantee of capital safety.

Models use an expanding training window from June 2022 onward and retrain every 42 bars (7 days). Each horizon leaves room for its forward labels to become observable. Sample weights decay with a 540-day half-life; observations in the top 5% of absolute forward returns have their weights reset to 1.

The main strategy reads precomputed thresholds from `data/{coin}_80D_wfa.csv`. If the file is missing, it falls back to a fixed threshold, `0.52` by default. `--threshold` controls only this fallback and does not override an existing WFA file.

## 3. Main Strategy: Volatility Turning Points

- **Entry:** The score passes its threshold, a turning point occurs, the current bar's price return is positive, and the short-horizon score meets its requirement.
- **Short-horizon confirmation:** The mean of the 1-, 3-, and 5-day probabilities must reach `0.54` for BTC or `0.50` for BNB.
- **Waiting queue:** A triggered entry with an insufficient short-horizon score can wait up to 12 bars (48 hours) for confirmation.
- **Exit:** The score falls below its threshold, or a turning point occurs with a negative price return on the current bar.

Regular entries and exits are simulated at the next bar's open. Positions still open at the end of the backtest are settled at the final close. The default path does not enable leverage amplification, volatility-based exposure limits, drawdown circuit breakers, or cooldowns.

## 4. Trend Module & Dynamic Blending

The trend module operates independently. It enters when 20-bar momentum is positive and the close breaks above the previous 20 bars' highest high; it exits when the close falls below the previous 10 bars' lowest low. This module is not subject to the main strategy's score gate and also uses 1x exposure.

The two modules' returns are blended each bar. The trend module's weight is:

```text
w_trend = 0.40 / (1 + exp(-0.3 * Z_MA200))
w_main  = 1 - w_trend
```

`Z_MA200` is the current close's percentage deviation from the 200-day moving average computed through the previous bar. The trend module receives at most 40%; the main strategy receives at least 60%.

## 5. BTC / BNB Rotation

Rank the BTC and BNB **strategy equity curves** by their returns over the past 120 bars (20 days). The stronger strategy receives 70% and the other 30%, with weights applied one bar later. Rankings use strategy performance, not asset price returns.

Portfolio equity is synthesized from weighted return series. The current implementation does not deduct additional costs for portfolio rebalancing.

## 6. Out-of-Sample Live Validation

**Validation period: March 1 to September 18, 2026.** The author reports that the strategy was validated in live trading during this period under the following protocol:

- Strategy rules were fixed before the validation period began.
- Scheduled model and threshold updates used only historical information available at each decision point.
- Subsequent market outcomes provided the forward, out-of-sample evaluation of those decisions.

This is a walk-forward protocol: earlier observations from the validation period can enter later training windows once they become known. It does not mean that the entire period was permanently excluded from all subsequent training.

The README's table and chart are a backtest reproduction of this window, not a live-account statement. The repository does not include the live fills or account equity records needed to independently reconcile realized returns. The backtest assumptions and timing limitations below still apply to the reproduced figures.

## 7. Data Updates & Reproduction

The bundled price and DVOL data end at **2026-09-18 16:00** (CSV timestamp). The README's statistics and chart cover March 1, 2026 through that timestamp and were verified by rerunning the repository's code.

```bash
python backtest.py --start 2026-03-01 --end 2026-09-18T16:00 --vs-hold
```

`fetch_data.py` updates prices and DVOL through Binance and Deribit public endpoints without an API key. It combines coarser historical DVOL with finer recent data, resamples to 4 hours, and forward-fills gaps. Some historical values are therefore approximations.

**The update script does not regenerate WFA thresholds.** The bundled threshold tables end on May 21, 2026 for BTC and April 28, 2026 for BNB. Later bars reuse the last available threshold. The threshold-generation pipeline is not included in this repository.

## 8. Statistics & Limitations

- Trading modules include a fee of `0.07%` per side. Funding, additional slippage, liquidity impact, and portfolio rebalancing costs are not modeled.
- Maximum drawdown is measured from portfolio equity. The Sharpe ratio uses daily returns, annualized by the square root of 365, without subtracting a risk-free rate.
- Trade counts and win rates aggregate the modules' trade records; they do not represent actual orders placed by a portfolio account.
- Dynamic blending uses the current bar's closing information to weight that same bar's module returns. A full audit of timing and live execution feasibility is outside the scope of this backtest.
- The sample covers only the stated period and BTC/BNB. Data approximations, precomputed thresholds, and execution assumptions affect results. Past performance does not guarantee future returns.

## Tests

```bash
pip install pytest
pytest -v
```

Tests cover feature calculations, the trading loop, the trend module, rotation, and statistics. See the [contributing guide](../CONTRIBUTING.md) for details.
