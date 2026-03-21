from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .math import ParticleFilterConfig, compute_particle_filter_frame, compute_regime_frame
from .math.features import compute_market_features
from .math.kelly import kelly_fraction_binary
from .math.model import predict_up_probability
from .strategy import apply_particle_filter_entry_filter, decide_trade_side


@dataclass(frozen=True)
class BacktestConfig:
    initial_bankroll: float = 1000.0
    market_up_price: float = 0.50
    market_down_price: float = 0.50
    fee_rate: float = 0.0156
    edge_buffer: float = 0.02
    fractional_kelly: float = 0.50
    max_fraction: float = 0.20
    min_history: int = 40
    entry_timing: str = "close"
    use_regime_filter: bool = True
    regime_lookback: int = 480
    regime_min_history: int = 80
    use_particle_filter: bool = True
    particle_filter_lookback: int = 240
    particle_filter_particles: int = 300
    particle_filter_entry_filter: bool = True
    particle_filter_min_gap: float = 0.0
    walk_forward_indicators: bool = True
    regime_refit_every: int = 96
    particle_filter_refit_every: int = 8


@dataclass(frozen=True)
class BacktestStats:
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_return_pct: float
    final_bankroll: float
    max_drawdown_pct: float
    avg_trade_return_pct: float


def run_backtest(
    candles_15m: pd.DataFrame,
    config: BacktestConfig,
    odds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    entry_timing = config.entry_timing.lower()
    if entry_timing not in {"close", "open"}:
        raise ValueError("entry_timing must be either 'close' or 'open'")

    required = config.min_history + (2 if entry_timing == "close" else 1)
    if len(candles_15m) < required:
        raise ValueError("Not enough candles for backtest")

    df = candles_15m.sort_index().copy()
    interval = _infer_candle_interval(df.index)
    bankroll = float(config.initial_bankroll)

    if odds_df is not None and not odds_df.empty:
        odds = odds_df.copy()
        if "timestamp" in odds.columns:
            odds["timestamp"] = pd.to_datetime(odds["timestamp"], utc=True)
            odds = odds.set_index("timestamp")
        odds = odds.sort_index()
    else:
        odds = pd.DataFrame(index=df.index)

    rows: list[dict[str, object]] = []
    regime_frame, pf_frame = _build_indicator_frames(df=df, config=config)

    start_i = config.min_history
    end_i = len(df) - 1 if entry_timing == "close" else len(df)

    for i in range(start_i, end_i):
        ts = df.index[i]
        if entry_timing == "close":
            hist = df.iloc[: i + 1]
            live_price = float(df.iloc[i]["close"])
            entry_price = float(df.iloc[i]["close"])
            exit_price = float(df.iloc[i + 1]["close"])
            window_end = df.index[i + 1]
        else:
            hist = df.iloc[:i]
            live_price = float(df.iloc[i]["open"])
            entry_price = float(df.iloc[i]["open"])
            exit_price = float(df.iloc[i]["close"])
            window_end = df.index[i + 1] if (i + 1) < len(df) else (ts + interval)

        market_up = _lookup_odds(odds, ts, "up_price", config.market_up_price)
        market_down = _lookup_odds(odds, ts, "down_price", config.market_down_price)

        order_book_proxy = {
            "depth_imbalance": 0.0,
            "wall_skew_log": 0.0,
            "top_wall_distance_skew": 0.0,
        }
        features = compute_market_features(
            candles_15m=hist,
            live_price=live_price,
            order_book_features=order_book_proxy,
            funding_zscore=0.0,
            liquidation_imbalance=0.0,
            news_shock=0.0,
        )

        model_out = predict_up_probability(features)
        p_up = model_out.p_up
        regime_ts = ts if entry_timing == "close" else df.index[max(i - 1, 0)]
        regime_row = _lookup_regime(regime_frame, regime_ts)
        pf_row = _lookup_indicator(pf_frame, regime_ts)
        regime_name = str(regime_row.get("regime", "unfiltered"))
        allowed_side = str(regime_row.get("allowed_side", "BOTH"))
        regime_confidence = float(regime_row.get("confidence", 0.0))
        decision = decide_trade_side(
            p_up=p_up,
            market_up_price=market_up,
            market_down_price=market_down,
            edge_buffer=config.edge_buffer,
            regime=regime_name,
            allowed_side=allowed_side,
        )
        if config.use_particle_filter and config.particle_filter_entry_filter:
            decision = apply_particle_filter_entry_filter(
                decision=decision,
                observed_price=live_price,
                fair_price=float(pf_row.get("pf_fair_price", np.nan)),
                min_gap=config.particle_filter_min_gap,
            )

        stake_fraction = 0.0
        stake_usd = 0.0
        pnl = 0.0
        trade_ret_pct = 0.0
        kelly_raw_fraction = 0.0
        kelly_edge = 0.0
        kelly_break_even_prob = 0.0
        effective_share_price = 0.0
        p_side = 0.0
        side_price = 0.0
        outcome_side = "UP" if exit_price >= entry_price else "DOWN"

        if decision.side in {"UP", "DOWN"}:
            p_side = float(p_up if decision.side == "UP" else (1.0 - p_up))
            side_price = float(market_up if decision.side == "UP" else market_down)
            kelly = kelly_fraction_binary(
                p_win=p_side,
                share_price=side_price,
                fee_rate=config.fee_rate,
                fractional_kelly=config.fractional_kelly,
                max_fraction=config.max_fraction,
            )
            stake_fraction = float(kelly.fraction)
            kelly_raw_fraction = float(kelly.raw_kelly)
            kelly_edge = float(kelly.edge)
            kelly_break_even_prob = float(kelly.break_even_prob)
            effective_share_price = float(kelly.effective_share_price)
            stake_usd = bankroll * stake_fraction

            if stake_usd > 0:
                shares = stake_usd / effective_share_price
                payout = shares if decision.side == outcome_side else 0.0
                pnl = payout - stake_usd
                trade_ret_pct = (pnl / stake_usd) if stake_usd > 0 else 0.0
                bankroll += pnl

        rows.append(
            {
                "timestamp": ts,
                "window_end": window_end,
                "entry_timing": entry_timing,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "close_now": entry_price,
                "close_next": exit_price,
                "outcome_side": outcome_side,
                "decision": decision.side,
                "decision_reason": decision.reason,
                "regime": regime_name,
                "regime_allowed_side": allowed_side,
                "regime_confidence": regime_confidence,
                "blocked_by_regime": bool(decision.blocked_by_regime),
                "blocked_by_particle_filter": bool(decision.blocked_by_particle_filter),
                "bull_prob": float(regime_row.get("bull_prob", np.nan)),
                "neutral_prob": float(regime_row.get("neutral_prob", np.nan)),
                "bear_prob": float(regime_row.get("bear_prob", np.nan)),
                "pf_fair_price": float(pf_row.get("pf_fair_price", np.nan)),
                "pf_gap": float(pf_row.get("pf_gap", np.nan)),
                "pf_gap_pct": float(pf_row.get("pf_gap_pct", np.nan)),
                "pf_drift": float(pf_row.get("pf_drift", np.nan)),
                "pf_uncertainty": float(pf_row.get("pf_uncertainty", np.nan)),
                "pf_confidence": float(pf_row.get("pf_confidence", np.nan)),
                "pf_price_below_fair": bool(pf_row.get("pf_price_below_fair", False)),
                "pf_price_above_fair": bool(pf_row.get("pf_price_above_fair", False)),
                "pf_bull_long_setup": bool(pf_row.get("pf_bull_long_setup", False)),
                "pf_bear_short_setup": bool(pf_row.get("pf_bear_short_setup", False)),
                "p_up": p_up,
                "p_side": p_side,
                "side_price": side_price,
                "edge_up": decision.edge_up,
                "edge_down": decision.edge_down,
                "market_up_price": market_up,
                "market_down_price": market_down,
                "kelly_raw_fraction": kelly_raw_fraction,
                "kelly_raw_fraction_pct": kelly_raw_fraction * 100.0,
                "kelly_edge": kelly_edge,
                "kelly_break_even_prob": kelly_break_even_prob,
                "effective_share_price": effective_share_price,
                "stake_fraction": stake_fraction,
                "stake_fraction_pct": stake_fraction * 100.0,
                "stake_usd": stake_usd,
                "pnl_usd": pnl,
                "trade_return_pct": trade_ret_pct,
                "bankroll": bankroll,
            }
        )

    out = pd.DataFrame(rows)
    return out


def summarize_backtest(results: pd.DataFrame, initial_bankroll: float) -> BacktestStats:
    if results.empty:
        return BacktestStats(
            trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            total_return_pct=0.0,
            final_bankroll=initial_bankroll,
            max_drawdown_pct=0.0,
            avg_trade_return_pct=0.0,
        )

    traded = results[results["decision"].isin(["UP", "DOWN"])].copy()
    wins = int((traded["pnl_usd"] > 0).sum())
    losses = int((traded["pnl_usd"] < 0).sum())
    trades = int(len(traded))

    bankroll = results["bankroll"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(bankroll)
    drawdown = np.where(peaks > 0, (peaks - bankroll) / peaks, 0.0)

    final_bankroll = float(results.iloc[-1]["bankroll"])
    total_return_pct = (final_bankroll / initial_bankroll) - 1.0 if initial_bankroll > 0 else 0.0

    return BacktestStats(
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=(wins / trades) if trades > 0 else 0.0,
        total_return_pct=float(total_return_pct),
        final_bankroll=final_bankroll,
        max_drawdown_pct=float(drawdown.max()) if drawdown.size > 0 else 0.0,
        avg_trade_return_pct=float(traded["trade_return_pct"].mean()) if trades > 0 else 0.0,
    )


def _lookup_odds(odds: pd.DataFrame, ts: pd.Timestamp, col: str, fallback: float) -> float:
    if col not in odds.columns or odds.empty:
        return float(fallback)

    if ts in odds.index:
        val = odds.at[ts, col]
        return float(val) if np.isfinite(val) else float(fallback)

    subset = odds.loc[odds.index <= ts]
    if subset.empty:
        return float(fallback)

    val = subset.iloc[-1][col]
    return float(val) if np.isfinite(val) else float(fallback)


def _infer_candle_interval(index: pd.Index) -> pd.Timedelta:
    if len(index) >= 2:
        inferred = pd.Timedelta(index[1] - index[0])
        if inferred > pd.Timedelta(0):
            return inferred
    return pd.Timedelta(minutes=15)


def _lookup_regime(regime_frame: pd.DataFrame, ts: pd.Timestamp) -> dict[str, object]:
    return _lookup_indicator(regime_frame, ts)


def _lookup_indicator(frame: pd.DataFrame, ts: pd.Timestamp) -> dict[str, object]:
    if frame.empty:
        return {}
    if ts in frame.index:
        return frame.loc[ts].to_dict()

    subset = frame.loc[frame.index <= ts]
    if subset.empty:
        return frame.iloc[0].to_dict()
    return subset.iloc[-1].to_dict()


def _build_indicator_frames(
    df: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.walk_forward_indicators:
        regime_frame = pd.DataFrame()
        if config.use_regime_filter:
            regime_frame = compute_regime_frame(
                candles_15m=df,
                lookback=config.regime_lookback,
                min_history=config.regime_min_history,
            )
        pf_frame = pd.DataFrame()
        if config.use_particle_filter:
            pf_frame = compute_particle_filter_frame(
                candles_15m=df,
                regime_frame=regime_frame if not regime_frame.empty else None,
                config=ParticleFilterConfig(
                    num_particles=config.particle_filter_particles,
                    lookback=config.particle_filter_lookback,
                ),
            )
        return regime_frame, pf_frame

    regime_frame = _build_walk_forward_regime_frame(df=df, config=config) if config.use_regime_filter else pd.DataFrame()
    pf_frame = _build_walk_forward_particle_filter_frame(df=df, config=config, regime_frame=regime_frame) if config.use_particle_filter else pd.DataFrame()
    return regime_frame, pf_frame


def _build_walk_forward_regime_frame(
    df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    warmup = max(config.min_history, config.regime_min_history)
    step = max(1, config.regime_refit_every)
    rows: list[dict[str, object]] = []

    for end_i in _refit_points(length=len(df), warmup=warmup, step=step):
        hist = _history_slice(df=df, end_i=end_i, lookback=config.regime_lookback)
        frame = compute_regime_frame(
            candles_15m=hist,
            lookback=config.regime_lookback,
            min_history=config.regime_min_history,
        )
        if frame.empty:
            continue
        row = frame.iloc[-1].to_dict()
        row["timestamp"] = df.index[end_i]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def _build_walk_forward_particle_filter_frame(
    df: pd.DataFrame,
    config: BacktestConfig,
    regime_frame: pd.DataFrame,
) -> pd.DataFrame:
    warmup = max(config.min_history, config.regime_min_history)
    step = max(1, config.particle_filter_refit_every)
    rows: list[dict[str, object]] = []

    for end_i in _refit_points(length=len(df), warmup=warmup, step=step):
        hist = _history_slice(
            df=df,
            end_i=end_i,
            lookback=max(config.particle_filter_lookback, config.regime_lookback),
        )
        regime_hist = regime_frame.loc[regime_frame.index <= df.index[end_i]].copy() if not regime_frame.empty else pd.DataFrame()
        pf = compute_particle_filter_frame(
            candles_15m=hist,
            regime_frame=regime_hist if not regime_hist.empty else None,
            config=ParticleFilterConfig(
                num_particles=config.particle_filter_particles,
                lookback=config.particle_filter_lookback,
            ),
        )
        if pf.empty:
            continue
        row = pf.iloc[-1].to_dict()
        row["timestamp"] = df.index[end_i]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def _refit_points(length: int, warmup: int, step: int) -> list[int]:
    if length <= warmup:
        return []
    points = list(range(warmup, length, step))
    final_idx = length - 1
    if not points or points[-1] != final_idx:
        points.append(final_idx)
    return points


def _history_slice(df: pd.DataFrame, end_i: int, lookback: int) -> pd.DataFrame:
    if lookback <= 0:
        return df.iloc[: end_i + 1]
    start_i = max(0, end_i + 1 - lookback)
    return df.iloc[start_i : end_i + 1]
