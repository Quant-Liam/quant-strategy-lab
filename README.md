# BTC 15m Regime Console

This repo now centers the live BTC tracker around a simpler regime-gated workflow:

- live Coinbase BTC price on the main dashboard
- fixed 720 x 1-minute candles for the chart
- a 3-state Markov switching regime model fed by a GARCH volatility filter
- a regime-aware particle filter fair-value indicator
- bull regime = only `UP` trades
- bear regime = only `DOWN` trades
- neutral regime = no trade
- clearer Kelly sizing breakdown
- paper-trade logging with live BTC spot price at entry
- a reusable backtest engine and CLI

## Project Layout

- [/Users/liamrodgers/Desktop/Python/Personal/dashboard.py](/Users/liamrodgers/Desktop/Python/Personal/dashboard.py): Streamlit dashboard
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math): all model math in one place
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/features.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/features.py): price and market feature engineering
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/model.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/model.py): probability model and feature contributions
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/kelly.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/kelly.py): Kelly sizing math
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py): Markov switching plus GARCH regime model
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py): regime-aware particle filter fair-value indicator
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/strategy.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/strategy.py): trade-direction rules and regime gating
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/paper.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/paper.py): paper-trade logging and settlement
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py): historical simulation engine
- [/Users/liamrodgers/Desktop/Python/Personal/run_backtest.py](/Users/liamrodgers/Desktop/Python/Personal/run_backtest.py): CLI backtest runner

Top-level `btc15m/features.py`, `btc15m/model.py`, and `btc15m/kelly.py` remain as compatibility shims, but the actual math now lives under `btc15m/math/`.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the dashboard:

```bash
streamlit run dashboard.py
```

3. In the sidebar:

- use `Trade Action` at the top to log or settle paper trades
- set `UP` and `DOWN` market prices
- tune Kelly settings
- optionally open `External Signals` for funding, liquidation, and news inputs

## Dashboard Notes

- the old order-book wall map was removed to keep the interface cleaner
- the price chart always uses `720` one-minute candles
- the dashboard still uses Coinbase live price and order-book data under the hood
- logged paper trades now capture both the prediction market price and live BTC spot price
- Kelly is shown as a full setup table so the break-even point, raw Kelly, candidate size, and applied size are visible
- the particle filter fair price is plotted on the main price chart and `price - PF fair value` is visible as both a metric and a lower panel

## Regime Model

The regime engine lives in [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py).

It does two things:

- fits a simple GARCH(1,1) conditional volatility filter on 15-minute log returns
- fits a 3-state Markov switching model on volatility-standardized returns

The resulting states are labeled `bull`, `neutral`, and `bear` from their weighted return characteristics.

Trading rules:

- `bull`: only `UP` trades are allowed
- `bear`: only `DOWN` trades are allowed
- `neutral`: no trades are allowed

## Particle Filter Indicator

The fair-value filter lives in [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py).

It is intentionally not a second regime classifier. It uses the current MS-GARCH regime as context and estimates:

- PF fair price
- PF drift
- PF uncertainty and confidence
- `real price - PF fair price`

Useful hooks exposed in code:

- `is_price_below_pf_fair_value(...)`
- `is_price_above_pf_fair_value(...)`
- `get_pf_gap(...)`

Typical usage:

- `bull` regime: look for price below PF fair price
- `bear` regime: look for price above PF fair price

## Paper Trading

The paper-trade log keeps:

- direction and market share price
- live BTC spot price at entry
- regime, allowed side, and regime confidence
- Kelly sizing, stake, settlement, and PnL

Existing CSV logs are auto-upgraded with the new columns when the dashboard starts.

## Backtesting

CLI example:

```bash
python run_backtest.py \
  --years 2 \
  --granularity 900 \
  --entry-timing open \
  --market-up-price 0.50 \
  --market-down-price 0.50 \
  --output-csv backtest_results.csv \
  --distribution-csv backtest_distribution.csv
```

Useful regime options:

- `--disable-regime-filter`: turn off the bull/bear/neutral trade gate
- `--regime-lookback`: candles used for regime fitting, with `0` meaning fit once on the full series
- `--regime-min-history`: minimum candles before the regime model is considered active

Useful particle-filter options:

- `--disable-particle-filter`: turn off the fair-value indicator in backtests
- `--particle-filter-lookback`: candles used by the particle filter, with `0` meaning full-series
- `--particle-filter-particles`: particle count for the fair-value estimate

## Dependencies

Current requirements file:

- `numpy`
- `pandas`
- `plotly`
- `requests`
- `scipy`
- `statsmodels`
- `streamlit`
- `streamlit-autorefresh`
- `python-dotenv`
