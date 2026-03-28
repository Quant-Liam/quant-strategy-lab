from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .math import ParticleFilterConfig, compute_kelly_from_pf, compute_particle_filter_frame, compute_regime_frame


@dataclass(frozen=True)
class BacktestConfig:
    market_up_price: float = 0.50
    market_down_price: float = 0.50
    fee_rate: float = 0.0156
    alpha: float = 1.5
    min_gap_scale: float = 0.001
    fractional_kelly: float = 0.50
    max_fraction: float = 0.20
    regime_confidence_threshold: float = 0.45
    particle_filter_particles: int = 300
    use_confidence_shrink: bool = True
    min_normalized_gap: float = 0.8
    min_raw_gap: float = 100.0
    max_pf_confidence: float = 0.4
    use_regime_filter: bool = False
    max_regime_confidence: float = 0.8
    use_pwin_filter: bool = False
    min_p_win: float = 0.6
    allowed_hours: tuple[int, ...] | None = None
    allowed_days: tuple[int, ...] | None = None
    log_filter_reasons: bool = True


def normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame sorted by candle start time."""

    if candles.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = candles.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"candles are missing required columns: {missing}")

    numeric_cols = [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    frame = frame.sort_index().drop_duplicates(keep="last")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame[["open", "high", "low", "close", "volume"]]


def build_backtest_signal_frame(
    candles_15m: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Join OHLC candles with full-history regime and particle-filter state."""

    active_config = config or BacktestConfig()
    candles = normalize_candles(candles_15m)
    if candles.empty:
        return candles

    lookback = max(len(candles), 81)
    regime_frame = compute_regime_frame(candles, lookback=lookback)
    pf_frame = compute_particle_filter_frame(
        candles_15m=candles,
        regime_frame=regime_frame,
        config=ParticleFilterConfig(
            num_particles=active_config.particle_filter_particles,
            lookback=lookback,
        ),
    )
    pf_columns = [
        "pf_fair_price",
        "pf_gap",
        "pf_gap_pct",
        "pf_drift",
        "pf_uncertainty",
        "pf_confidence",
        "pf_price_below_fair",
        "pf_price_above_fair",
        "pf_bull_long_setup",
        "pf_bear_short_setup",
    ]
    return candles.join(regime_frame, how="left").join(pf_frame[pf_columns], how="left")


def evaluate_trade_filters(
    trade_row: pd.Series,
    min_normalized_gap: float = 0.8,
    min_raw_gap: float = 100.0,
    max_pf_confidence: float = 0.4,
    use_regime_filter: bool = False,
    max_regime_confidence: float = 0.8,
    use_pwin_filter: bool = False,
    min_p_win: float = 0.6,
    allowed_hours: Sequence[int] | None = None,
    allowed_days: Sequence[int] | None = None,
) -> tuple[bool, list[str]]:
    """Return whether a baseline trade passes the high-edge filter layer."""

    failures: list[str] = []

    normalized_gap = _finite_or_nan(trade_row.get("normalized_gap"))
    raw_gap = _finite_or_nan(trade_row.get("raw_gap"))
    pf_confidence = _finite_or_nan(trade_row.get("pf_confidence"))
    regime_confidence = _finite_or_nan(trade_row.get("regime_confidence"))
    p_win = _finite_or_nan(trade_row.get("p_win"))
    entry_hour_utc = trade_row.get("entry_hour_utc")
    entry_weekday = trade_row.get("entry_weekday")

    if not np.isfinite(normalized_gap) or normalized_gap < float(min_normalized_gap):
        failures.append("normalized_gap_below_min")
    if not np.isfinite(raw_gap) or raw_gap < float(min_raw_gap):
        failures.append("raw_gap_below_min")
    if not np.isfinite(pf_confidence) or pf_confidence > float(max_pf_confidence):
        failures.append("pf_confidence_above_max")
    if use_regime_filter and (not np.isfinite(regime_confidence) or regime_confidence > float(max_regime_confidence)):
        failures.append("regime_confidence_above_max")
    if use_pwin_filter and (not np.isfinite(p_win) or p_win < float(min_p_win)):
        failures.append("p_win_below_min")
    if allowed_hours is not None and int(entry_hour_utc) not in set(int(hour) for hour in allowed_hours):
        failures.append("hour_not_allowed")
    if allowed_days is not None and int(entry_weekday) not in set(int(day) for day in allowed_days):
        failures.append("weekday_not_allowed")

    return len(failures) == 0, failures


def run_backtest(
    candles_15m: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Backtest interval-open entries with baseline and filtered trade cohorts."""

    active_config = config or BacktestConfig()
    signal_frame = build_backtest_signal_frame(candles_15m, config=active_config)
    if signal_frame.empty or len(signal_frame) < 2:
        return pd.DataFrame()

    bar_delta = _infer_bar_delta(signal_frame.index)
    rows: list[dict[str, object]] = []

    for row_number in range(1, len(signal_frame)):
        signal_ts = signal_frame.index[row_number - 1]
        entry_ts = signal_frame.index[row_number]
        signal_row = signal_frame.iloc[row_number - 1]
        entry_row = signal_frame.iloc[row_number]
        entry_price = float(entry_row["open"])
        exit_price = float(entry_row["close"])
        regime_label = str(signal_row.get("regime", "neutral")).lower()
        regime_confidence = _finite_or_none(signal_row.get("confidence"))
        pf_fair_price = _finite_or_none(signal_row.get("pf_fair_price"))
        pf_uncertainty = _finite_or_none(signal_row.get("pf_uncertainty"))
        pf_confidence = _finite_or_none(signal_row.get("pf_confidence"))
        market_share_price = _market_share_price_for_regime(active_config, regime_label)

        sizing = None
        if market_share_price is not None:
            sizing = compute_kelly_from_pf(
                live_price=entry_price,
                fair_price_pf=pf_fair_price if pf_fair_price is not None else np.nan,
                pf_uncertainty=pf_uncertainty if pf_uncertainty is not None else np.nan,
                pf_confidence=pf_confidence,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                market_share_price=market_share_price,
                fee_rate=active_config.fee_rate,
                alpha=active_config.alpha,
                min_gap_scale=active_config.min_gap_scale,
                fractional_kelly=active_config.fractional_kelly,
                max_fraction=active_config.max_fraction,
                use_confidence_shrink=active_config.use_confidence_shrink,
            )

        baseline_no_trade_reason = _baseline_no_trade_reason(
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            regime_confidence_threshold=active_config.regime_confidence_threshold,
            sizing=sizing,
        )
        baseline_trade_taken = int(baseline_no_trade_reason is None)
        trade_side = sizing.trade_side if sizing is not None else None
        win_loss = _win_loss_for_trade(trade_side, entry_price, exit_price) if baseline_trade_taken else np.nan

        row = {
            "signal_time_utc": signal_ts,
            "entry_time_utc": entry_ts,
            "interval_end_utc": entry_ts + bar_delta,
            "interval_start_price": entry_price,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "trade_side": trade_side,
            "regime": regime_label,
            "regime_confidence": regime_confidence,
            "pf_fair_price": pf_fair_price,
            "pf_uncertainty": pf_uncertainty,
            "pf_confidence": pf_confidence,
            "raw_gap": sizing.raw_gap if sizing is not None else np.nan,
            "normalized_gap": sizing.normalized_gap if sizing is not None else np.nan,
            "z_score": sizing.z_score if sizing is not None else np.nan,
            "p_base": sizing.p_base if sizing is not None else np.nan,
            "p_win": sizing.p_final if sizing is not None else np.nan,
            "confidence_multiplier": sizing.confidence_multiplier if sizing is not None else np.nan,
            "market_share_price": market_share_price,
            "kelly_fraction": sizing.kelly_fraction if sizing is not None else 0.0,
            "kelly_size_pct": (sizing.kelly_fraction * 100.0) if sizing is not None else 0.0,
            "raw_kelly": sizing.raw_kelly if sizing is not None else 0.0,
            "expected_log_growth": sizing.expected_log_growth if sizing is not None else 0.0,
            "break_even_prob": sizing.break_even_prob if sizing is not None else np.nan,
            "effective_share_price": sizing.effective_share_price if sizing is not None else np.nan,
            "net_odds": sizing.net_odds if sizing is not None else np.nan,
            "entry_hour_utc": int(entry_ts.hour),
            "entry_weekday": int(entry_ts.weekday()),
            "baseline_trade_taken": baseline_trade_taken,
            "baseline_no_trade_reason": baseline_no_trade_reason,
            "win_loss": win_loss,
        }

        passed_filters = False
        filter_failures: list[str] = []
        if baseline_trade_taken:
            passed_filters, filter_failures = evaluate_trade_filters(
                pd.Series(row),
                min_normalized_gap=active_config.min_normalized_gap,
                min_raw_gap=active_config.min_raw_gap,
                max_pf_confidence=active_config.max_pf_confidence,
                use_regime_filter=active_config.use_regime_filter,
                max_regime_confidence=active_config.max_regime_confidence,
                use_pwin_filter=active_config.use_pwin_filter,
                min_p_win=active_config.min_p_win,
                allowed_hours=active_config.allowed_hours,
                allowed_days=active_config.allowed_days,
            )

        row["filter_passed"] = int(passed_filters)
        row["filtered_trade_taken"] = int(baseline_trade_taken and passed_filters)
        row["filtered_no_trade_reason"] = "" if passed_filters else ",".join(filter_failures)
        row["filter_failure_reasons"] = ",".join(filter_failures) if active_config.log_filter_reasons else ""
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Compare filtered and baseline cohorts in a compact one-row summary."""

    if results.empty:
        return pd.DataFrame()

    baseline = results.loc[results["baseline_trade_taken"] == 1].copy()
    filtered = results.loc[results["filtered_trade_taken"] == 1].copy()

    baseline_count = int(len(baseline))
    filtered_count = int(len(filtered))
    baseline_win_rate = float(baseline["win_loss"].mean()) if baseline_count else np.nan
    filtered_win_rate = float(filtered["win_loss"].mean()) if filtered_count else np.nan
    reduction = np.nan
    if baseline_count:
        reduction = float(100.0 * (1.0 - (filtered_count / baseline_count)))

    summary = {
        "total_trades_filtered": filtered_count,
        "win_rate_filtered": filtered_win_rate,
        "total_trades_unfiltered": baseline_count,
        "win_rate_unfiltered": baseline_win_rate,
        "trade_frequency_reduction_pct": reduction,
        "wins_filtered": int(filtered["win_loss"].sum()) if filtered_count else 0,
        "wins_unfiltered": int(baseline["win_loss"].sum()) if baseline_count else 0,
        "avg_kelly_fraction_filtered": float(filtered["kelly_fraction"].mean()) if filtered_count else np.nan,
        "avg_kelly_fraction_unfiltered": float(baseline["kelly_fraction"].mean()) if baseline_count else np.nan,
    }
    return pd.DataFrame([summary])


def run_filter_grid_search(
    results: pd.DataFrame,
    normalized_gap_thresholds: Sequence[float],
    raw_gap_thresholds: Sequence[float],
    pf_confidence_thresholds: Sequence[float],
    *,
    use_regime_filter: bool = False,
    max_regime_confidence: float = 0.8,
    use_pwin_filter: bool = False,
    min_p_win: float = 0.6,
    allowed_hours: Sequence[int] | None = None,
    allowed_days: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Sweep filter thresholds against the baseline cohort and rank outcomes."""

    if results.empty:
        return pd.DataFrame()

    baseline = results.loc[results["baseline_trade_taken"] == 1].copy()
    baseline_count = int(len(baseline))
    baseline_win_rate = float(baseline["win_loss"].mean()) if baseline_count else np.nan
    if baseline.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for normalized_gap, raw_gap, pf_confidence in product(
        normalized_gap_thresholds,
        raw_gap_thresholds,
        pf_confidence_thresholds,
    ):
        filtered_mask = _vectorized_filter_mask(
            baseline,
            min_normalized_gap=float(normalized_gap),
            min_raw_gap=float(raw_gap),
            max_pf_confidence=float(pf_confidence),
            use_regime_filter=use_regime_filter,
            max_regime_confidence=max_regime_confidence,
            use_pwin_filter=use_pwin_filter,
            min_p_win=min_p_win,
            allowed_hours=allowed_hours,
            allowed_days=allowed_days,
        )
        cohort = baseline.loc[filtered_mask].copy()
        trade_count = int(len(cohort))
        win_rate = float(cohort["win_loss"].mean()) if trade_count else np.nan
        reduction = float(100.0 * (1.0 - (trade_count / baseline_count))) if baseline_count else np.nan
        rows.append(
            {
                "min_normalized_gap": float(normalized_gap),
                "min_raw_gap": float(raw_gap),
                "max_pf_confidence": float(pf_confidence),
                "use_regime_filter": int(use_regime_filter),
                "max_regime_confidence": float(max_regime_confidence),
                "use_pwin_filter": int(use_pwin_filter),
                "min_p_win": float(min_p_win),
                "total_trades_filtered": trade_count,
                "win_rate_filtered": win_rate,
                "total_trades_unfiltered": baseline_count,
                "win_rate_unfiltered": baseline_win_rate,
                "trade_frequency_reduction_pct": reduction,
            }
        )

    sweep = pd.DataFrame(rows)
    if not sweep.empty:
        sweep = sweep.sort_values(
            by=["win_rate_filtered", "total_trades_filtered", "min_normalized_gap", "min_raw_gap"],
            ascending=[False, False, True, True],
            na_position="last",
        ).reset_index(drop=True)
    return sweep


def plot_filter_sweep_tradeoff(
    sweep_results: pd.DataFrame,
    threshold_column: str,
):
    """Plot average win rate and trade count against one swept parameter."""

    if sweep_results.empty:
        raise ValueError("sweep_results is empty")
    if threshold_column not in sweep_results.columns:
        raise ValueError(f"{threshold_column} is not a column in sweep_results")

    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("plotly is required to plot sweep tradeoffs") from exc

    grouped = (
        sweep_results.groupby(threshold_column, dropna=False)
        .agg(
            win_rate_filtered=("win_rate_filtered", "mean"),
            total_trades_filtered=("total_trades_filtered", "mean"),
        )
        .reset_index()
        .sort_values(threshold_column)
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=grouped[threshold_column],
            y=grouped["win_rate_filtered"],
            mode="lines+markers",
            name="Win Rate",
            yaxis="y1",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grouped[threshold_column],
            y=grouped["total_trades_filtered"],
            mode="lines+markers",
            name="Trade Count",
            yaxis="y2",
        )
    )
    figure.update_layout(
        title=f"Filtered Win Rate vs Trade Count by {threshold_column}",
        xaxis_title=threshold_column,
        yaxis=dict(title="Average Win Rate", tickformat=".1%"),
        yaxis2=dict(title="Average Trade Count", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return figure


def export_backtest_excel(
    output_path: str | Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    sweep_results: pd.DataFrame | None = None,
) -> Path:
    """Write baseline, filtered, and summary views to a workbook."""

    workbook_path = Path(output_path)
    baseline = results.loc[results["baseline_trade_taken"] == 1].copy() if not results.empty else pd.DataFrame()
    filtered = results.loc[results["filtered_trade_taken"] == 1].copy() if not results.empty else pd.DataFrame()
    excel_summary = _excel_safe_frame(summary)
    excel_baseline = _excel_safe_frame(baseline)
    excel_filtered = _excel_safe_frame(filtered)
    excel_results = _excel_safe_frame(results)
    excel_sweep = _excel_safe_frame(sweep_results) if sweep_results is not None else None

    try:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            excel_summary.to_excel(writer, sheet_name="summary", index=False)
            excel_baseline.to_excel(writer, sheet_name="baseline_trades", index=False)
            excel_filtered.to_excel(writer, sheet_name="filtered_trades", index=False)
            excel_results.to_excel(writer, sheet_name="all_intervals", index=False)
            if excel_sweep is not None and not excel_sweep.empty:
                excel_sweep.to_excel(writer, sheet_name="grid_search", index=False)
    except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
        if exc.name == "openpyxl":
            raise RuntimeError(
                "openpyxl is required to export Excel files. Install it with "
                "`python -m pip install openpyxl` and rerun the backtest."
            ) from exc
        raise

    return workbook_path


def _baseline_no_trade_reason(
    *,
    regime_label: str,
    regime_confidence: float | None,
    regime_confidence_threshold: float,
    sizing,
) -> str | None:
    if regime_label not in {"bull", "bear"}:
        return "neutral_regime"
    if regime_confidence is None or not np.isfinite(regime_confidence):
        return "invalid_regime_confidence"
    if regime_confidence <= float(regime_confidence_threshold):
        return "regime_confidence_below_threshold"
    if sizing is None:
        return "missing_kelly_inputs"
    return sizing.no_trade_reason


def _market_share_price_for_regime(config: BacktestConfig, regime_label: str) -> float | None:
    if regime_label == "bull":
        return float(config.market_up_price)
    if regime_label == "bear":
        return float(config.market_down_price)
    return None


def _win_loss_for_trade(trade_side: str | None, entry_price: float, exit_price: float) -> int:
    if str(trade_side).upper() == "UP":
        return int(exit_price > entry_price)
    if str(trade_side).upper() == "DOWN":
        return int(exit_price < entry_price)
    return 0


def _vectorized_filter_mask(
    frame: pd.DataFrame,
    *,
    min_normalized_gap: float,
    min_raw_gap: float,
    max_pf_confidence: float,
    use_regime_filter: bool,
    max_regime_confidence: float,
    use_pwin_filter: bool,
    min_p_win: float,
    allowed_hours: Sequence[int] | None,
    allowed_days: Sequence[int] | None,
) -> pd.Series:
    mask = (
        frame["normalized_gap"].ge(float(min_normalized_gap))
        & frame["raw_gap"].ge(float(min_raw_gap))
        & frame["pf_confidence"].le(float(max_pf_confidence))
    )
    if use_regime_filter:
        mask &= frame["regime_confidence"].le(float(max_regime_confidence))
    if use_pwin_filter:
        mask &= frame["p_win"].ge(float(min_p_win))
    if allowed_hours is not None:
        mask &= frame["entry_hour_utc"].isin([int(hour) for hour in allowed_hours])
    if allowed_days is not None:
        mask &= frame["entry_weekday"].isin([int(day) for day in allowed_days])
    return mask.fillna(False)


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _finite_or_nan(value: object) -> float:
    numeric = _finite_or_none(value)
    return float(numeric) if numeric is not None else float("nan")


def _infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=15)
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return pd.Timedelta(minutes=15)
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return pd.Timedelta(minutes=15)
    return median_delta


def _excel_safe_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()

    safe = frame.copy()
    for column in safe.columns:
        series = safe[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            safe[column] = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return safe
