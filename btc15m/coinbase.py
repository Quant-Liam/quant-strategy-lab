from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

COINBASE_BASE_URL = "https://api.exchange.coinbase.com"
ALLOWED_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}


@dataclass(frozen=True)
class OrderBookSnapshot:
    bids: pd.DataFrame
    asks: pd.DataFrame
    best_bid: float
    best_ask: float
    mid: float
    sequence: int | None
    received_at: pd.Timestamp


class CoinbaseClient:
    def __init__(self, base_url: str = COINBASE_BASE_URL, timeout: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "btc-15m-research/1.0",
            }
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def fetch_live_price(self, product_id: str = "BTC-USD") -> float:
        data = self._get(f"/products/{product_id}/ticker")
        return float(data["price"])

    def fetch_candles(
        self,
        product_id: str = "BTC-USD",
        granularity: int = 900,
        limit: int = 300,
    ) -> pd.DataFrame:
        if granularity not in ALLOWED_GRANULARITIES:
            raise ValueError(f"granularity must be one of {sorted(ALLOWED_GRANULARITIES)}")
        if limit <= 0:
            raise ValueError("limit must be > 0")

        rows: list[list[Any]] = []

        if limit <= 300:
            payload = self._get(
                f"/products/{product_id}/candles",
                params={"granularity": granularity},
            )
            rows.extend(payload)
        else:
            end = pd.Timestamp.utcnow().floor(f"{granularity}s")
            remaining = limit
            while remaining > 0:
                chunk = min(300, remaining)
                start = end - pd.Timedelta(seconds=granularity * chunk)
                payload = self._get(
                    f"/products/{product_id}/candles",
                    params={
                        "granularity": granularity,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                    },
                )
                rows.extend(payload)
                remaining -= chunk
                end = start

        if not rows:
            raise RuntimeError("Coinbase returned no candle data")

        frame = pd.DataFrame(
            rows,
            columns=["timestamp", "low", "high", "open", "close", "volume"],
        )
        numeric_cols = ["low", "high", "open", "close", "volume"]
        frame[numeric_cols] = frame[numeric_cols].astype(float)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        frame = (
            frame.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .set_index("timestamp")
            .tail(limit)
        )
        return frame

    def fetch_order_book(self, product_id: str = "BTC-USD", level: int = 2) -> OrderBookSnapshot:
        if level not in {1, 2, 3}:
            raise ValueError("level must be 1, 2, or 3")

        payload = self._get(
            f"/products/{product_id}/book",
            params={"level": level},
        )

        bids = _parse_book_side(payload.get("bids", []))
        asks = _parse_book_side(payload.get("asks", []))

        if bids.empty or asks.empty:
            raise RuntimeError("Coinbase order book returned empty bids or asks")

        bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
        asks = asks.sort_values("price", ascending=True).reset_index(drop=True)

        best_bid = float(bids.iloc[0]["price"])
        best_ask = float(asks.iloc[0]["price"])

        return OrderBookSnapshot(
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=(best_bid + best_ask) / 2.0,
            sequence=int(payload["sequence"]) if payload.get("sequence") else None,
            received_at=pd.Timestamp.now(tz="UTC"),
        )


def _parse_book_side(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["price", "size", "num_orders"])

    trimmed = [row[:3] for row in rows]
    frame = pd.DataFrame(trimmed, columns=["price", "size", "num_orders"])
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
    frame["num_orders"] = pd.to_numeric(frame["num_orders"], errors="coerce")
    frame = frame.dropna(subset=["price", "size"])
    return frame


def compute_order_book_features(
    snapshot: OrderBookSnapshot,
    band_bps: int = 50,
) -> dict[str, float]:
    if band_bps <= 0:
        raise ValueError("band_bps must be > 0")

    bids = snapshot.bids.copy()
    asks = snapshot.asks.copy()
    bids["notional"] = bids["price"] * bids["size"]
    asks["notional"] = asks["price"] * asks["size"]

    band = snapshot.mid * (band_bps / 10_000.0)
    bids_near = bids[bids["price"] >= (snapshot.mid - band)]
    asks_near = asks[asks["price"] <= (snapshot.mid + band)]

    bid_notional = float(bids_near["notional"].sum())
    ask_notional = float(asks_near["notional"].sum())
    total_notional = bid_notional + ask_notional

    depth_imbalance = 0.0
    if total_notional > 0:
        depth_imbalance = (bid_notional - ask_notional) / total_notional

    top_bid_wall_notional = 0.0
    top_bid_wall_distance_bps = 0.0
    if not bids_near.empty:
        bid_wall_row = bids_near.loc[bids_near["notional"].idxmax()]
        top_bid_wall_notional = float(bid_wall_row["notional"])
        top_bid_wall_distance_bps = float((snapshot.mid - bid_wall_row["price"]) / snapshot.mid * 10_000)

    top_ask_wall_notional = 0.0
    top_ask_wall_distance_bps = 0.0
    if not asks_near.empty:
        ask_wall_row = asks_near.loc[asks_near["notional"].idxmax()]
        top_ask_wall_notional = float(ask_wall_row["notional"])
        top_ask_wall_distance_bps = float((ask_wall_row["price"] - snapshot.mid) / snapshot.mid * 10_000)

    wall_skew_log = float(np.log((top_bid_wall_notional + 1.0) / (top_ask_wall_notional + 1.0)))

    return {
        "best_bid": float(snapshot.best_bid),
        "best_ask": float(snapshot.best_ask),
        "mid": float(snapshot.mid),
        "spread_bps": float((snapshot.best_ask - snapshot.best_bid) / snapshot.mid * 10_000),
        "band_bps": float(band_bps),
        "bid_notional_near": bid_notional,
        "ask_notional_near": ask_notional,
        "depth_imbalance": depth_imbalance,
        "top_bid_wall_notional": top_bid_wall_notional,
        "top_ask_wall_notional": top_ask_wall_notional,
        "top_bid_wall_distance_bps": top_bid_wall_distance_bps,
        "top_ask_wall_distance_bps": top_ask_wall_distance_bps,
        "top_wall_distance_skew": float(top_ask_wall_distance_bps - top_bid_wall_distance_bps),
        "wall_skew_log": wall_skew_log,
    }


def order_book_ladder(
    snapshot: OrderBookSnapshot,
    band_bps: int = 80,
    levels: int = 30,
) -> pd.DataFrame:
    if levels <= 0:
        raise ValueError("levels must be > 0")

    mid = snapshot.mid
    band = mid * (band_bps / 10_000.0)

    bids = snapshot.bids[snapshot.bids["price"] >= mid - band].copy()
    asks = snapshot.asks[snapshot.asks["price"] <= mid + band].copy()

    bids = bids.sort_values("price", ascending=False).head(levels)
    asks = asks.sort_values("price", ascending=True).head(levels)

    bids["side"] = "bid"
    asks["side"] = "ask"

    frame = pd.concat([bids, asks], ignore_index=True)
    frame["notional"] = frame["price"] * frame["size"]
    frame["distance_bps"] = (frame["price"] - mid) / mid * 10_000
    return frame
