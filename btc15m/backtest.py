from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import compute_market_features
from .kelly import kelly_fraction_binary
from .model import predict_up_probability
from .strategy import decide_trade_side


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
        decision = decide_trade_side(
            p_up=p_up,
            market_up_price=market_up,
            market_down_price=market_down,
            edge_buffer=config.edge_buffer,
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
