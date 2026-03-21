from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings

try:
    import pandas as pd
    from statsmodels.tools.sm_exceptions import ValueWarning
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
from run_backtest import build_distribution_table, normalize_candles, summarize_regimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Easy terminal runner for the 2-year BTC regime + particle-filter backtest."
    )
    parser.add_argument("--product-id", default="BTC-USD")
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--granularity", type=int, default=900)
    parser.add_argument("--candles-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("pf_regime_backtest_results.csv"))
    parser.add_argument("--distribution-csv", type=Path, default=Path("pf_regime_backtest_distribution.csv"))
    parser.add_argument("--initial-bankroll", type=float, default=1000.0)
    parser.add_argument("--market-up-price", type=float, default=0.50)
    parser.add_argument("--market-down-price", type=float, default=0.50)
    parser.add_argument("--fee-rate", type=float, default=0.0156)
    parser.add_argument("--edge-buffer", type=float, default=0.02)
    parser.add_argument("--fractional-kelly", type=float, default=0.50)
    parser.add_argument("--max-fraction", type=float, default=0.20)
    parser.add_argument("--min-history", type=int, default=40)
    parser.add_argument("--regime-lookback", type=int, default=480)
    parser.add_argument("--regime-min-history", type=int, default=80)
    parser.add_argument("--regime-refit-every", type=int, default=288)
    parser.add_argument("--particle-filter-lookback", type=int, default=240)
    parser.add_argument("--particle-filter-particles", type=int, default=300)
    parser.add_argument("--particle-filter-refit-every", type=int, default=16)
    parser.add_argument("--particle-filter-min-gap", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=10)
    return parser.parse_args()


def estimate_candle_count(years: float, granularity: int) -> int:
    seconds = years * 365.25 * 24 * 60 * 60
    return int((seconds + granularity - 1) // granularity)


def load_candles(args: argparse.Namespace) -> pd.DataFrame:
    if args.candles_csv is not None:
        frame = pd.read_csv(args.candles_csv)
        return normalize_candles(frame)

    bars = estimate_candle_count(args.years, args.granularity)
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
    return normalize_candles(frame)


def main() -> None:
    warnings.simplefilter("ignore", ValueWarning)
    args = parse_args()
    candles_15m = load_candles(args)

    config = BacktestConfig(
        initial_bankroll=args.initial_bankroll,
        market_up_price=args.market_up_price,
        market_down_price=args.market_down_price,
        fee_rate=args.fee_rate,
        edge_buffer=args.edge_buffer,
        fractional_kelly=args.fractional_kelly,
        max_fraction=args.max_fraction,
        min_history=args.min_history,
        entry_timing="open",
        use_regime_filter=True,
        regime_lookback=args.regime_lookback,
        regime_min_history=args.regime_min_history,
        use_particle_filter=True,
        particle_filter_lookback=args.particle_filter_lookback,
        particle_filter_particles=args.particle_filter_particles,
        particle_filter_entry_filter=True,
        particle_filter_min_gap=args.particle_filter_min_gap,
        walk_forward_indicators=True,
        regime_refit_every=args.regime_refit_every,
        particle_filter_refit_every=args.particle_filter_refit_every,
    )

    results = run_backtest(candles_15m=candles_15m, config=config)
    stats = summarize_backtest(results=results, initial_bankroll=config.initial_bankroll)
    regime_summary = summarize_regimes(results)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.distribution_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    build_distribution_table(results, stats).to_csv(args.distribution_csv, index=False)

    pf_blocked = int(results["blocked_by_particle_filter"].fillna(False).astype(bool).sum())
    bull_summary = regime_summary.get("bull", {"bars": 0, "episodes": 0})
    bear_summary = regime_summary.get("bear", {"bars": 0, "episodes": 0})

    print("PF regime backtest complete")
    print(f"Window: {candles_15m.index.min()} -> {candles_15m.index.max()}")
    print(f"Rows: {len(results)} | Trades: {stats.trades} | Win rate: {stats.win_rate * 100:.2f}%")
    print(f"PF-blocked entries: {pf_blocked}")
    print(f"Bull regime bars: {bull_summary['bars']} | Bull regime episodes: {bull_summary['episodes']}")
    print(f"Bear regime bars: {bear_summary['bars']} | Bear regime episodes: {bear_summary['episodes']}")
    print(f"Final bankroll: ${stats.final_bankroll:,.2f} | Total return: {stats.total_return_pct * 100:.2f}%")
    print(f"Results CSV: {args.output_csv.resolve()}")
    print(f"Distribution CSV: {args.distribution_csv.resolve()}")


if __name__ == "__main__":
    main()
