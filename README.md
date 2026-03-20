# BTC 15m Coinbase Decision Console

This repo now includes a live dashboard that pulls Coinbase BTC market data and computes:

- Live price + 15-minute window start price
- Coinbase order-book wall/depth metrics (bid vs ask pressure)
- A probabilistic UP/DOWN signal model
- Kelly-based position sizing from your model probability and market odds
- Automatic external signals for funding, liquidations, and news shock
- Paper-trade logging with auto settlement and performance summary
- A reusable backtest engine + notebook for historical strategy tests

## Files

- `/Users/liamrodgers/Desktop/Python/Personal/dashboard.py`: Streamlit live frontend
- `/Users/liamrodgers/Desktop/Python/Personal/btc15m/coinbase.py`: Coinbase client + order-book wall features
- `/Users/liamrodgers/Desktop/Python/Personal/btc15m/features.py`: 15m feature engineering
- `/Users/liamrodgers/Desktop/Python/Personal/btc15m/model.py`: probability model + feature contributions
- `/Users/liamrodgers/Desktop/Python/Personal/btc15m/kelly.py`: Kelly sizing
- `/Users/liamrodgers/Desktop/Python/Personal/btc15m/strategy.py`: trade/no-trade decision logic
- `/Users/liamrodgers/Desktop/Python/Personal/btc15m/external.py`: funding/liquidation/news adapters
- `/Users/liamrodgers/Desktop/Python/Personal/btc15m/paper.py`: paper-trade log + settlement + summary
- `/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py`: historical simulation engine
- `/Users/liamrodgers/Desktop/Python/Personal/BTC15m_backtest.ipynb`: backtest notebook

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run dashboard:

```bash
streamlit run dashboard.py
```

3. In the sidebar, set:

- `UP share price` and `DOWN share price` from your market
- fee assumptions
- fractional Kelly and max bankroll at risk
- auto/manual funding, liquidation, and news signals
- paper-trading log file and bankroll

## Notes On Signals

- `order-book walls`: computed directly from Coinbase level-2 book snapshots.
- `funding z-score`: auto-fetched from Binance funding history and normalized as a z-score.
- `liquidation imbalance`: auto-fetched from Binance forced liquidations and computed as `(short_liq - long_liq) / total`.
- `news shock`: auto-fetched from CryptoCompare headlines, scored by keyword sentiment and recency weighting.

If any auto feed fails, the dashboard automatically falls back to your manual input.

## Paper Trading

Use the `Paper Trading` section in the dashboard sidebar:

- `Log current recommendation`: stores the current decision, probability, edge, and Kelly stake in `paper_trades.csv`.
- `Auto settle open paper trades`: marks trades as settled once the 15-minute window ends, and computes realized PnL.
- `Settle open trades now`: forces settlement check immediately.

The dashboard shows:

- total/open/settled trade counts
- win rate
- total paper PnL
- recent trade log table

## Backtesting

Open `/Users/liamrodgers/Desktop/Python/Personal/BTC15m_backtest.ipynb` and run all cells.

This notebook:

- pulls historical Coinbase 15m candles
- runs the model + decision + Kelly sizing loop
- computes summary stats (win rate, return, max drawdown)
- plots equity curve and trade-return distribution

Optional: provide an `ODDS_CSV_PATH` with columns `timestamp,up_price,down_price` to replay historical market odds instead of fixed odds.

### CLI Backtest (2-Year, 15m, CSV Output)

Run this from `/Users/liamrodgers/Desktop/Python/Personal`:

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

This command:

- uses 15-minute BTC candles
- places each trade at candle open (`--entry-timing open`) to reduce lookahead bias
- uses neutral market odds (0.50 / 0.50) by default
- exports:
  - `backtest_results.csv`: interval-level results including `pnl_usd`, `stake_fraction_pct`, and `kelly_raw_fraction_pct`
  - `backtest_distribution.csv`: aggregate and quantile statistics for distribution analysis

## Risk Controls To Add Before Any Live Trading

- Hard daily loss stop (for example: stop after -3R or -5% bankroll)
- Max concurrent risk cap
- API latency and stale-data checks
- Kill switch when data streams fail
- Paper-trading mode and full trade logging
