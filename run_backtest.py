from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from btc15m import BacktestConfig, CoinbaseClient, run_backtest, summarize_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC 15m strategy backtests and export CSV outputs.")
    parser.add_argument("--product-id", default="BTC-USD", help="Coinbase product id (default: BTC-USD).")
    parser.add_argument(
        "--granularity",
        type=int,
        default=900,
        help="Candle size in seconds. Use 900 for 15-minute candles.",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=2.0,
        help="Lookback window in years when pulling candles from Coinbase.",
    )
    parser.add_argument(
        "--candles-csv",
        type=Path,
        default=None,
        help="Optional CSV path for historical candles instead of downloading.",
    )
    parser.add_argument(
        "--odds-csv",
        type=Path,
        default=None,
        help="Optional CSV path with timestamp,up_price,down_price.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("backtest_results.csv"),
        help="Per-interval backtest output CSV path.",
    )
    parser.add_argument(
        "--distribution-csv",
        type=Path,
        default=Path("backtest_distribution.csv"),
        help="Distribution/statistics CSV path.",
    )
    parser.add_argument("--initial-bankroll", type=float, default=1000.0)
    parser.add_argument("--market-up-price", type=float, default=0.50)
    parser.add_argument("--market-down-price", type=float, default=0.50)
    parser.add_argument("--fee-rate", type=float, default=0.0156)
    parser.add_argument("--edge-buffer", type=float, default=0.02)
    parser.add_argument("--fractional-kelly", type=float, default=0.50)
    parser.add_argument("--max-fraction", type=float, default=0.20)
    parser.add_argument("--min-history", type=int, default=40)
    parser.add_argument(
        "--entry-timing",
        choices=["open", "close"],
        default="open",
        help="Trade timing: open (start of candle) or close (end of candle).",
    )
    parser.add_argument("--timeout", type=int, default=10, help="Coinbase request timeout in seconds.")
    return parser.parse_args()


def estimate_candle_count(years: float, granularity: int) -> int:
    if years <= 0:
        raise ValueError("years must be > 0")
    if granularity <= 0:
        raise ValueError("granularity must be > 0")
    seconds = years * 365.25 * 24 * 60 * 60
    return int(math.ceil(seconds / granularity))


def load_candles(args: argparse.Namespace) -> pd.DataFrame:
    if args.candles_csv is not None:
        frame = pd.read_csv(args.candles_csv)
    else:
        bars = estimate_candle_count(years=args.years, granularity=args.granularity)
        print(
            f"Fetching {bars} candles for {args.product_id} "
            f"({args.years:.2f} years at {args.granularity}s granularity)..."
        )
        client = CoinbaseClient(timeout=args.timeout)
        frame = client.fetch_candles(
            product_id=args.product_id,
            granularity=args.granularity,
            limit=bars,
        )

    candles = normalize_candles(frame)
    return candles


def normalize_candles(frame: pd.DataFrame) -> pd.DataFrame:
    candles = frame.copy()

    candles.columns = [str(c).strip().lower() for c in candles.columns]

    if "timestamp" in candles.columns:
        candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
        candles = candles.set_index("timestamp")

    if not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError("Candles must have a DatetimeIndex or a timestamp column")

    if candles.index.tz is None:
        candles.index = candles.index.tz_localize("UTC")
    else:
        candles.index = candles.index.tz_convert("UTC")

    required_cols = ["open", "high", "low", "close"]
    missing = [c for c in required_cols if c not in candles.columns]
    if missing:
        raise ValueError(f"Candles are missing required columns: {missing}")

    num_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in candles.columns]
    candles[num_cols] = candles[num_cols].apply(pd.to_numeric, errors="coerce")
    candles = candles.dropna(subset=["open", "high", "low", "close"])

    candles = candles.sort_index().drop_duplicates(keep="last")
    return candles


def load_odds(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    odds_df = pd.read_csv(path)
    return odds_df


def build_distribution_table(results: pd.DataFrame, stats) -> pd.DataFrame:
    traded = results[results["decision"].isin(["UP", "DOWN"])].copy()

    rows: list[dict[str, object]] = [
        {"section": "overall", "group": "all", "metric": "rows", "value": float(len(results))},
        {"section": "overall", "group": "all", "metric": "trades", "value": float(stats.trades)},
        {"section": "overall", "group": "all", "metric": "wins", "value": float(stats.wins)},
        {"section": "overall", "group": "all", "metric": "losses", "value": float(stats.losses)},
        {"section": "overall", "group": "all", "metric": "win_rate", "value": float(stats.win_rate)},
        {
            "section": "overall",
            "group": "all",
            "metric": "total_return_pct",
            "value": float(stats.total_return_pct * 100.0),
        },
        {
            "section": "overall",
            "group": "all",
            "metric": "final_bankroll",
            "value": float(stats.final_bankroll),
        },
        {
            "section": "overall",
            "group": "all",
            "metric": "max_drawdown_pct",
            "value": float(stats.max_drawdown_pct * 100.0),
        },
        {
            "section": "overall",
            "group": "all",
            "metric": "avg_trade_return_pct",
            "value": float(stats.avg_trade_return_pct * 100.0),
        },
    ]

    if traded.empty:
        return pd.DataFrame(rows)

    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    series_specs = [
        ("pnl_usd", "pnl_usd"),
        ("trade_return_pct", "trade_return_pct"),
        ("stake_fraction_pct", "stake_fraction_pct"),
        ("kelly_raw_fraction_pct", "kelly_raw_fraction_pct"),
    ]

    for column, metric_name in series_specs:
        for q in quantiles:
            value = float(traded[column].quantile(q))
            rows.append(
                {
                    "section": "quantile",
                    "group": "all",
                    "metric": f"{metric_name}_q{int(q * 100):02d}",
                    "value": value,
                }
            )

    for side in ["UP", "DOWN"]:
        side_df = traded[traded["decision"] == side]
        if side_df.empty:
            continue
        rows.extend(
            [
                {
                    "section": "by_side",
                    "group": side,
                    "metric": "trades",
                    "value": float(len(side_df)),
                },
                {
                    "section": "by_side",
                    "group": side,
                    "metric": "win_rate",
                    "value": float((side_df["pnl_usd"] > 0).mean()),
                },
                {
                    "section": "by_side",
                    "group": side,
                    "metric": "avg_pnl_usd",
                    "value": float(side_df["pnl_usd"].mean()),
                },
                {
                    "section": "by_side",
                    "group": side,
                    "metric": "total_pnl_usd",
                    "value": float(side_df["pnl_usd"].sum()),
                },
                {
                    "section": "by_side",
                    "group": side,
                    "metric": "avg_stake_fraction_pct",
                    "value": float(side_df["stake_fraction_pct"].mean()),
                },
            ]
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    candles_15m = load_candles(args)
    odds_df = load_odds(args.odds_csv)

    config = BacktestConfig(
        initial_bankroll=args.initial_bankroll,
        market_up_price=args.market_up_price,
        market_down_price=args.market_down_price,
        fee_rate=args.fee_rate,
        edge_buffer=args.edge_buffer,
        fractional_kelly=args.fractional_kelly,
        max_fraction=args.max_fraction,
        min_history=args.min_history,
        entry_timing=args.entry_timing,
    )

    results = run_backtest(
        candles_15m=candles_15m,
        config=config,
        odds_df=odds_df,
    )
    stats = summarize_backtest(results=results, initial_bankroll=config.initial_bankroll)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.distribution_csv.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(args.output_csv, index=False)
    distribution = build_distribution_table(results, stats)
    distribution.to_csv(args.distribution_csv, index=False)

    print("Backtest complete")
    print(f"Window: {candles_15m.index.min()} -> {candles_15m.index.max()}")
    print(f"Rows: {len(results)} | Trades: {stats.trades} | Win rate: {stats.win_rate * 100:.2f}%")
    print(f"Final bankroll: ${stats.final_bankroll:,.2f} | Total return: {stats.total_return_pct * 100:.2f}%")
    print(f"Results CSV: {args.output_csv.resolve()}")
    print(f"Distribution CSV: {args.distribution_csv.resolve()}")


if __name__ == "__main__":
    main()
