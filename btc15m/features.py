from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def current_15m_window(ts: datetime | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if ts is None:
        ts = datetime.now(timezone.utc)
    minute = (ts.minute // 15) * 15
    start = ts.replace(minute=minute, second=0, microsecond=0)
    end = start + pd.Timedelta(minutes=15)
    return pd.Timestamp(start), pd.Timestamp(end)


def infer_window_start_price(candles_1m: pd.DataFrame, window_start: pd.Timestamp, fallback_price: float) -> float:
    if candles_1m.empty:
        return float(fallback_price)

    if candles_1m.index.tz is None:
        candles_1m = candles_1m.tz_localize("UTC")

    if window_start.tzinfo is None:
        window_start = window_start.tz_localize("UTC")

    exact = candles_1m.loc[candles_1m.index == window_start]
    if not exact.empty:
        return float(exact.iloc[-1]["open"])

    before = candles_1m.loc[candles_1m.index < window_start]
    if not before.empty:
        return float(before.iloc[-1]["close"])

    after = candles_1m.loc[candles_1m.index > window_start]
    if not after.empty:
        return float(after.iloc[0]["open"])

    return float(fallback_price)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def compute_price_features(candles_15m: pd.DataFrame, live_price: float) -> dict[str, float]:
    if len(candles_15m) < 20:
        raise ValueError("Need at least 20 x 15-minute candles to compute features")

    closes = candles_15m["close"].astype(float)
    highs = candles_15m["high"].astype(float)
    lows = candles_15m["low"].astype(float)

    log_close = np.log(closes)
    rets = log_close.diff().dropna()

    mom_30m = float(log_close.iloc[-1] - log_close.iloc[-3])
    mom_60m = float(log_close.iloc[-1] - log_close.iloc[-5])
    mom_120m = float(log_close.iloc[-1] - log_close.iloc[-9])

    vol_2h = float(rets.tail(8).std())
    range_2h = float(np.log(highs.tail(8).max() / lows.tail(8).min()))
    recent_30m_move = float(abs(log_close.iloc[-1] - log_close.iloc[-3]))

    stalling = 1.0 if (abs(mom_60m) > 0.003 and recent_30m_move < 0.0012) else 0.0
    live_vs_lastclose = float(np.log(float(live_price) / closes.iloc[-1]))

    rsi_val = float(_rsi(closes, period=14).iloc[-1])
    rsi_centered = (rsi_val - 50.0) / 50.0

    return {
        "mom_30m": mom_30m,
        "mom_60m": mom_60m,
        "mom_120m": mom_120m,
        "vol_2h": vol_2h,
        "range_2h": range_2h,
        "stalling": stalling,
        "live_vs_lastclose": live_vs_lastclose,
        "rsi_centered": float(rsi_centered),
    }


def compute_market_features(
    candles_15m: pd.DataFrame,
    live_price: float,
    order_book_features: dict[str, float],
    funding_zscore: float = 0.0,
    liquidation_imbalance: float = 0.0,
    news_shock: float = 0.0,
) -> dict[str, float]:
    price_features = compute_price_features(candles_15m=candles_15m, live_price=live_price)

    return {
        **price_features,
        "depth_imbalance": float(order_book_features.get("depth_imbalance", 0.0)),
        "wall_skew_log": float(order_book_features.get("wall_skew_log", 0.0)),
        "top_wall_distance_skew": float(order_book_features.get("top_wall_distance_skew", 0.0)),
        "funding_zscore": float(funding_zscore),
        "liquidation_imbalance": float(liquidation_imbalance),
        "news_shock": float(news_shock),
    }
