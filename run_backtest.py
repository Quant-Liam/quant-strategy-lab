from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    missing = exc.name or "a required package"
    print(
        "Missing Python dependency: "
        f"{missing}\n"
        "Install the project requirements, then rerun the backtest:\n"
        "python3 -m pip install -r /Users/liamrodgers/Desktop/Python/Personal/requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

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
        "--disable-regime-filter",
        action="store_true",
        help="Disable the bull/up, bear/down, neutral/no-trade regime gate.",
    )
    parser.add_argument(
        "--regime-lookback",
        type=int,
        default=480,
        help="15m candles used for the Markov-GARCH regime fit.",
    )
    parser.add_argument(
        "--regime-min-history",
        type=int,
        default=80,
        help="Minimum 15m candles required before the regime model activates.",
    )
    parser.add_argument(
        "--disable-particle-filter",
        action="store_true",
        help="Disable the regime-aware particle filter indicator during backtests.",
    )
    parser.add_argument(
        "--particle-filter-lookback",
        type=int,
        default=240,
        help="15m candles used for the particle filter.",
    )
    parser.add_argument(
        "--particle-filter-particles",
        type=int,
        default=300,
        help="Number of particles used by the fair-value filter.",
    )
    parser.add_argument(
        "--disable-particle-filter-entry-filter",
        action="store_true",
        help="Keep computing the PF indicator but do not require it for trade entry.",
    )
    parser.add_argument(
        "--particle-filter-min-gap",
        type=float,
        default=0.0,
        help="Minimum absolute price gap required versus PF fair value before entry is allowed.",
    )
    parser.add_argument(
        "--disable-walk-forward-indicators",
        action="store_true",
        help="Use full-sample indicator fitting instead of sparse walk-forward refits.",
    )
    parser.add_argument(
        "--regime-refit-every",
        type=int,
        default=96,
        help="How many 15m bars between walk-forward regime refits. Default 96 = daily.",
    )
    parser.add_argument(
        "--particle-filter-refit-every",
        type=int,
        default=8,
        help="How many 15m bars between walk-forward particle-filter refits. Default 8 = 2 hours.",
    )
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
    pf_blocked = results[results.get("blocked_by_particle_filter", False).astype(bool)].copy() if "blocked_by_particle_filter" in results.columns else pd.DataFrame()
    regime_summary = summarize_regimes(results)

    rows: list[dict[str, object]] = [
        {"section": "overall", "group": "all", "metric": "rows", "value": float(len(results))},
        {"section": "overall", "group": "all", "metric": "trades", "value": float(stats.trades)},
        {"section": "overall", "group": "all", "metric": "pf_blocked", "value": float(len(pf_blocked))},
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

    for regime_name, summary in regime_summary.items():
        rows.extend(
            [
                {
                    "section": "regime",
                    "group": regime_name,
                    "metric": "bars",
                    "value": float(summary["bars"]),
                },
                {
                    "section": "regime",
                    "group": regime_name,
                    "metric": "episodes",
                    "value": float(summary["episodes"]),
                },
            ]
        )

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


def summarize_regimes(results: pd.DataFrame) -> dict[str, dict[str, int]]:
    if "regime" not in results.columns or results.empty:
        return {}

    regime_series = results["regime"].fillna("unknown").astype(str)
    summary: dict[str, dict[str, int]] = {}
    for regime_name in ["bull", "bear", "neutral"]:
        mask = regime_series.eq(regime_name)
        bars = int(mask.sum())
        episodes = int((mask & ~mask.shift(1, fill_value=False)).sum())
        summary[regime_name] = {"bars": bars, "episodes": episodes}
    return summary


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
        use_regime_filter=not args.disable_regime_filter,
        regime_lookback=args.regime_lookback,
        regime_min_history=args.regime_min_history,
        use_particle_filter=not args.disable_particle_filter,
        particle_filter_lookback=args.particle_filter_lookback,
        particle_filter_particles=args.particle_filter_particles,
        particle_filter_entry_filter=not args.disable_particle_filter_entry_filter,
        particle_filter_min_gap=args.particle_filter_min_gap,
        walk_forward_indicators=not args.disable_walk_forward_indicators,
        regime_refit_every=args.regime_refit_every,
        particle_filter_refit_every=args.particle_filter_refit_every,
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
    if "blocked_by_particle_filter" in results.columns:
        pf_blocked = int(results["blocked_by_particle_filter"].fillna(False).astype(bool).sum())
        print(f"PF-blocked entries: {pf_blocked}")
    regime_summary = summarize_regimes(results)
    bull_summary = regime_summary.get("bull", {"bars": 0, "episodes": 0})
    bear_summary = regime_summary.get("bear", {"bars": 0, "episodes": 0})
    print(f"Bull regime bars: {bull_summary['bars']} | Bull regime episodes: {bull_summary['episodes']}")
    print(f"Bear regime bars: {bear_summary['bars']} | Bear regime episodes: {bear_summary['episodes']}")
    print(f"Final bankroll: ${stats.final_bankroll:,.2f} | Total return: {stats.total_return_pct * 100:.2f}%")
    print(f"Results CSV: {args.output_csv.resolve()}")
    print(f"Distribution CSV: {args.distribution_csv.resolve()}")


if __name__ == "__main__":
    main()
