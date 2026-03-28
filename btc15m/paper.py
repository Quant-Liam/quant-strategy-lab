from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


PAPER_COLUMNS = [
    "trade_id",
    "created_at_utc",
    "product_id",
    "window_start_utc",
    "window_end_utc",
    "side",
    "regime",
    "regime_allowed_side",
    "regime_confidence",
    "market_price",
    "spot_price",
    "fee_rate",
    "p_up",
    "p_down",
    "p_side",
    "edge",
    "kelly_fraction",
    "stake_fraction",
    "bankroll_usd",
    "stake_usd",
    "start_price",
    "end_price",
    "outcome_side",
    "status",
    "settled_at_utc",
    "pnl_usd",
    "return_pct",
    "notes",
]


@dataclass(frozen=True)
class PaperSummary:
    total: int
    open_count: int
    settled_count: int
    wins: int
    losses: int
    win_rate: float
    net_price_move: float
    avg_return_pct: float



def ensure_log(path: str | Path) -> Path:
    log_path = Path(path)
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        pd.DataFrame(columns=PAPER_COLUMNS).to_csv(log_path, index=False)
        return log_path

    frame = pd.read_csv(log_path)
    if list(frame.columns) != PAPER_COLUMNS:
        for column in PAPER_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame.reindex(columns=PAPER_COLUMNS)
        frame.to_csv(log_path, index=False)

    return log_path


def append_trade(path: str | Path, record: dict[str, object]) -> str:
    log_path = ensure_log(path)

    row = {col: record.get(col, None) for col in PAPER_COLUMNS}
    trade_id = str(row.get("trade_id") or uuid.uuid4())
    row["trade_id"] = trade_id

    frame = pd.DataFrame([row], columns=PAPER_COLUMNS)
    frame.to_csv(log_path, mode="a", index=False, header=False)
    return trade_id


def settle_open_trades(
    path: str | Path,
    candles_1m: pd.DataFrame,
    now_utc: pd.Timestamp | None = None,
) -> tuple[int, int]:
    log_path = ensure_log(path)
    frame = pd.read_csv(log_path)
    if frame.empty:
        return 0, 0

    for column in ("outcome_side", "status", "settled_at_utc"):
        frame[column] = frame[column].astype(object)

    if now_utc is None:
        now_utc = pd.Timestamp.now(tz="UTC")

    if candles_1m.index.tz is None:
        candles_1m = candles_1m.tz_localize("UTC")

    open_mask = frame["status"].fillna("OPEN").astype(str).str.upper().eq("OPEN")
    if not open_mask.any():
        return int(len(frame)), 0

    settled = 0

    for idx in frame.index[open_mask]:
        window_end = _parse_timestamp(frame.at[idx, "window_end_utc"])
        if window_end is None or window_end > now_utc:
            continue

        window_start = _parse_timestamp(frame.at[idx, "window_start_utc"])
        if window_start is None:
            continue

        start_price = _to_float(frame.at[idx, "start_price"], default=np.nan)
        if not np.isfinite(start_price) or start_price <= 0:
            start_price = _price_at_or_nearest(candles_1m, window_start)

        end_price = _price_at_or_nearest(candles_1m, window_end)
        if not np.isfinite(end_price) or not np.isfinite(start_price):
            continue

        outcome_side = "UP" if end_price >= start_price else "DOWN"

        side = str(frame.at[idx, "side"]).upper()
        raw_change = end_price - start_price

        pnl = np.nan
        ret = np.nan
        if side == "UP":
            pnl = raw_change
            ret = raw_change / start_price if start_price > 0 else np.nan
        elif side == "DOWN":
            pnl = -raw_change
            ret = (-raw_change) / start_price if start_price > 0 else np.nan

        frame.at[idx, "start_price"] = float(start_price)
        frame.at[idx, "end_price"] = float(end_price)
        frame.at[idx, "outcome_side"] = outcome_side
        frame.at[idx, "status"] = "SETTLED"
        frame.at[idx, "settled_at_utc"] = now_utc.isoformat()
        frame.at[idx, "pnl_usd"] = float(pnl) if np.isfinite(pnl) else np.nan
        frame.at[idx, "return_pct"] = float(ret) if np.isfinite(ret) else np.nan
        settled += 1

    frame.to_csv(log_path, index=False)
    return int(len(frame)), settled


def load_trades(path: str | Path) -> pd.DataFrame:
    log_path = ensure_log(path)
    frame = pd.read_csv(log_path)
    return frame


def summarize(path: str | Path) -> PaperSummary:
    frame = load_trades(path)
    if frame.empty:
        return PaperSummary(
            total=0,
            open_count=0,
            settled_count=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            net_price_move=0.0,
            avg_return_pct=0.0,
        )

    status = frame["status"].fillna("OPEN").astype(str).str.upper()
    settled = frame[status == "SETTLED"].copy()

    wins = int((settled["pnl_usd"].fillna(0) > 0).sum())
    losses = int((settled["pnl_usd"].fillna(0) < 0).sum())
    settled_count = int(len(settled))

    return PaperSummary(
        total=int(len(frame)),
        open_count=int((status == "OPEN").sum()),
        settled_count=settled_count,
        wins=wins,
        losses=losses,
        win_rate=(wins / settled_count) if settled_count > 0 else 0.0,
        net_price_move=float(settled["pnl_usd"].fillna(0.0).sum()) if settled_count > 0 else 0.0,
        avg_return_pct=float(settled["return_pct"].fillna(0.0).mean()) if settled_count > 0 else 0.0,
    )


def _parse_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _price_at_or_nearest(candles_1m: pd.DataFrame, ts: pd.Timestamp) -> float:
    exact = candles_1m.loc[candles_1m.index == ts]
    if not exact.empty:
        return float(exact.iloc[-1]["close"])

    before = candles_1m.loc[candles_1m.index <= ts]
    if not before.empty:
        return float(before.iloc[-1]["close"])

    after = candles_1m.loc[candles_1m.index > ts]
    if not after.empty:
        return float(after.iloc[0]["open"])

    return float("nan")
