from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

BINANCE_FAPI_BASE = "https://fapi.binance.com"
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"


class ExternalSignalError(RuntimeError):
    pass


def _get(url: str, params: dict[str, str | int] | None = None, timeout: int = 10) -> list[dict] | dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_binance_funding_zscore(symbol: str = "BTCUSDT", lookback: int = 100) -> float:
    if lookback < 30:
        raise ValueError("lookback should be >= 30")

    payload = _get(
        f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate",
        params={"symbol": symbol, "limit": min(1000, lookback)},
        timeout=10,
    )

    rates = np.array([float(row["fundingRate"]) for row in payload], dtype=float)
    if rates.size < 10:
        return 0.0

    mean = float(rates.mean())
    std = float(rates.std())
    if std < 1e-12:
        return 0.0

    zscore = (float(rates[-1]) - mean) / std
    return float(np.clip(zscore, -4.0, 4.0))


def fetch_binance_liquidation_imbalance(
    symbol: str = "BTCUSDT",
    lookback_hours: int = 24,
    limit: int = 1000,
) -> float:
    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be > 0")

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=lookback_hours)
    params = {
        "symbol": symbol,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(now.timestamp() * 1000),
        "limit": min(max(limit, 50), 1000),
    }

    payload = _get(f"{BINANCE_FAPI_BASE}/fapi/v1/allForceOrders", params=params, timeout=10)
    if not isinstance(payload, list) or len(payload) == 0:
        return 0.0

    long_liq_usd = 0.0
    short_liq_usd = 0.0

    for row in payload:
        side = str(row.get("side", "")).upper()

        qty = _to_float(row.get("origQty"), default=0.0)
        if qty <= 0:
            qty = _to_float(row.get("executedQty"), default=0.0)

        price = _to_float(row.get("avgPrice"), default=0.0)
        if price <= 0:
            price = _to_float(row.get("price"), default=0.0)

        notional = qty * price
        if notional <= 0:
            continue

        if side == "SELL":
            long_liq_usd += notional
        elif side == "BUY":
            short_liq_usd += notional

    total = long_liq_usd + short_liq_usd
    if total <= 0:
        return 0.0

    imbalance = (short_liq_usd - long_liq_usd) / total
    return float(np.clip(imbalance, -1.0, 1.0))


def fetch_cryptocompare_news_shock(
    lookback_minutes: int = 30,
    max_items: int = 50,
    categories: str = "BTC,Market",
    lang: str = "EN",
) -> float:
    if lookback_minutes <= 0:
        raise ValueError("lookback_minutes must be > 0")

    payload = _get(
        CRYPTOCOMPARE_NEWS_URL,
        params={"lang": lang, "categories": categories},
        timeout=10,
    )

    data = payload.get("Data", []) if isinstance(payload, dict) else []
    if not data:
        return 0.0

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    rows: list[dict] = []

    for item in data[:max_items]:
        published = item.get("published_on")
        if published is None:
            continue
        ts = datetime.fromtimestamp(int(published), tz=timezone.utc)
        if ts < cutoff:
            continue
        title = str(item.get("title", ""))
        body = str(item.get("body", ""))
        rows.append({"timestamp": ts, "text": f"{title} {body}".strip()})

    if not rows:
        return 0.0

    frame = pd.DataFrame(rows)
    frame["sentiment"] = frame["text"].map(_headline_sentiment)

    now = datetime.now(timezone.utc)
    age_minutes = (now - frame["timestamp"]).dt.total_seconds() / 60.0
    half_life = max(5.0, lookback_minutes / 2.0)
    weights = np.exp(-age_minutes / half_life)
    if float(weights.sum()) <= 0:
        return 0.0

    score = float((frame["sentiment"].to_numpy() * weights).sum() / weights.sum())
    return float(np.clip(score, -1.0, 1.0))


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _headline_sentiment(text: str) -> float:
    text_lower = text.lower()

    bullish_keywords = {
        "approval",
        "approved",
        "inflow",
        "rally",
        "surge",
        "breakout",
        "adoption",
        "partnership",
        "record high",
        "all-time high",
        "bullish",
        "buyback",
        "eases",
    }

    bearish_keywords = {
        "hack",
        "exploit",
        "lawsuit",
        "ban",
        "war",
        "attack",
        "liquidation",
        "outflow",
        "selloff",
        "bearish",
        "sanction",
        "tariff",
        "crash",
    }

    pos = sum(1 for term in bullish_keywords if term in text_lower)
    neg = sum(1 for term in bearish_keywords if term in text_lower)

    if pos == 0 and neg == 0:
        return 0.0

    raw = (pos - neg) / (pos + neg)
    return float(np.clip(raw, -1.0, 1.0))
