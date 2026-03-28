from __future__ import annotations

import argparse
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

from btc15m import CoinbaseClient
from btc15m.backtest import (
    BacktestConfig,
    export_backtest_excel,
    normalize_candles,
    run_backtest,
    run_filter_grid_search,
    summarize_backtest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BTC regime + PF + Kelly backtest and export an Excel workbook.")
    parser.add_argument("--product-id", default="BTC-USD")
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--granularity-15m", type=int, default=900)
    parser.add_argument("--candles-15m-csv", type=Path, default=None)
    parser.add_argument("--output-xlsx", type=Path, default=Path("pf_regime_backtest.xlsx"))
    parser.add_argument("--market-up-price", type=float, default=0.50)
    parser.add_argument("--market-down-price", type=float, default=0.50)
    parser.add_argument("--fee-rate", type=float, default=0.0156)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--min-gap-scale", type=float, default=0.001)
    parser.add_argument("--fractional-kelly", type=float, default=0.50)
    parser.add_argument("--max-fraction", type=float, default=0.20)
    parser.add_argument("--regime-confidence-threshold", type=float, default=0.45)
    parser.add_argument("--particle-filter-particles", type=int, default=300)
    parser.add_argument("--min-normalized-gap", type=float, default=0.8)
    parser.add_argument("--min-raw-gap", type=float, default=100.0)
    parser.add_argument("--max-pf-confidence", type=float, default=0.4)
    parser.add_argument("--use-regime-filter", action="store_true")
    parser.add_argument("--max-regime-confidence", type=float, default=0.8)
    parser.add_argument("--use-pwin-filter", action="store_true")
    parser.add_argument("--min-p-win", type=float, default=0.6)
    parser.add_argument("--allowed-hours", type=str, default=None, help="Comma-separated UTC hours, for example 0,1,2,13")
    parser.add_argument("--allowed-days", type=str, default=None, help="Comma-separated UTC weekdays, Monday=0 ... Sunday=6")
    parser.add_argument("--no-confidence-shrink", action="store_true")
    parser.add_argument("--no-filter-logging", action="store_true")
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--sweep-normalized-gap-grid", type=str, default="0.6,0.8,1.0,1.2")
    parser.add_argument("--sweep-raw-gap-grid", type=str, default="50,100,150,200")
    parser.add_argument("--sweep-pf-confidence-grid", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument("--timeout", type=int, default=10)
    return parser.parse_args()


def estimate_candle_count(years: float, granularity: int) -> int:
    seconds = years * 365.25 * 24 * 60 * 60
    return int((seconds + granularity - 1) // granularity)


def load_15m_candles(args: argparse.Namespace) -> pd.DataFrame:
    if args.candles_15m_csv is not None:
        return normalize_candles(pd.read_csv(args.candles_15m_csv))

    bars = estimate_candle_count(args.years, args.granularity_15m)
    print(
        f"Fetching {bars} x 15m candles for {args.product_id} "
        f"({args.years:.2f} years)..."
    )
    client = CoinbaseClient(timeout=args.timeout)
    return client.fetch_candles(
        product_id=args.product_id,
        granularity=args.granularity_15m,
        limit=bars,
    )


def parse_int_sequence(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_float_sequence(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def main() -> None:
    args = parse_args()
    candles_15m = load_15m_candles(args)

    config = BacktestConfig(
        market_up_price=args.market_up_price,
        market_down_price=args.market_down_price,
        fee_rate=args.fee_rate,
        alpha=args.alpha,
        min_gap_scale=args.min_gap_scale,
        fractional_kelly=args.fractional_kelly,
        max_fraction=args.max_fraction,
        regime_confidence_threshold=args.regime_confidence_threshold,
        particle_filter_particles=args.particle_filter_particles,
        use_confidence_shrink=not args.no_confidence_shrink,
        min_normalized_gap=args.min_normalized_gap,
        min_raw_gap=args.min_raw_gap,
        max_pf_confidence=args.max_pf_confidence,
        use_regime_filter=args.use_regime_filter,
        max_regime_confidence=args.max_regime_confidence,
        use_pwin_filter=args.use_pwin_filter,
        min_p_win=args.min_p_win,
        allowed_hours=parse_int_sequence(args.allowed_hours),
        allowed_days=parse_int_sequence(args.allowed_days),
        log_filter_reasons=not args.no_filter_logging,
    )

    results = run_backtest(candles_15m=candles_15m, config=config)
    summary = summarize_backtest(results)
    sweep_results = None
    if args.run_sweep:
        sweep_results = run_filter_grid_search(
            results,
            normalized_gap_thresholds=parse_float_sequence(args.sweep_normalized_gap_grid),
            raw_gap_thresholds=parse_float_sequence(args.sweep_raw_gap_grid),
            pf_confidence_thresholds=parse_float_sequence(args.sweep_pf_confidence_grid),
            use_regime_filter=args.use_regime_filter,
            max_regime_confidence=args.max_regime_confidence,
            use_pwin_filter=args.use_pwin_filter,
            min_p_win=args.min_p_win,
            allowed_hours=parse_int_sequence(args.allowed_hours),
            allowed_days=parse_int_sequence(args.allowed_days),
        )
    output_path = export_backtest_excel(args.output_xlsx, results, summary, sweep_results=sweep_results)

    baseline = results[results["baseline_trade_taken"] == 1].copy()
    filtered = results[results["filtered_trade_taken"] == 1].copy()
    print("Backtest complete")
    print(f"Window: {candles_15m.index.min()} -> {candles_15m.index.max()}")
    print(f"Saved workbook: {output_path.resolve()}")
    print(f"Baseline trades: {len(baseline)}")
    print(f"Filtered trades: {len(filtered)}")
    if not summary.empty:
        print(summary.to_string(index=False))
    if sweep_results is not None and not sweep_results.empty:
        print("\nTop grid-search rows:")
        print(sweep_results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
