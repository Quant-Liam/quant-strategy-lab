from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeDecision:
    side: str
    p_side: float
    market_price: float
    edge: float
    edge_up: float
    edge_down: float
    reason: str


def decide_trade_side(
    p_up: float,
    market_up_price: float,
    market_down_price: float,
    edge_buffer: float = 0.02,
) -> TradeDecision:
    if not 0 <= p_up <= 1:
        raise ValueError("p_up must be between 0 and 1")
    if not 0 < market_up_price < 1:
        raise ValueError("market_up_price must be between 0 and 1")
    if not 0 < market_down_price < 1:
        raise ValueError("market_down_price must be between 0 and 1")

    p_down = 1.0 - p_up
    edge_up = p_up - market_up_price
    edge_down = p_down - market_down_price

    if edge_up <= edge_buffer and edge_down <= edge_buffer:
        return TradeDecision(
            side="NO_TRADE",
            p_side=0.0,
            market_price=0.0,
            edge=max(edge_up, edge_down),
            edge_up=edge_up,
            edge_down=edge_down,
            reason="No edge above threshold",
        )

    if edge_up >= edge_down:
        return TradeDecision(
            side="UP",
            p_side=p_up,
            market_price=market_up_price,
            edge=edge_up,
            edge_up=edge_up,
            edge_down=edge_down,
            reason="UP has stronger model edge",
        )

    return TradeDecision(
        side="DOWN",
        p_side=p_down,
        market_price=market_down_price,
        edge=edge_down,
        edge_up=edge_up,
        edge_down=edge_down,
        reason="DOWN has stronger model edge",
    )
