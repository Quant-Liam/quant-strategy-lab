from __future__ import annotations

from dataclasses import dataclass

from .math.particle_filter import is_price_above_pf_fair_value, is_price_below_pf_fair_value


@dataclass(frozen=True)
class TradeDecision:
    side: str
    p_side: float
    market_price: float
    edge: float
    edge_up: float
    edge_down: float
    reason: str
    regime: str = "unfiltered"
    allowed_side: str = "BOTH"
    blocked_by_regime: bool = False
    blocked_by_particle_filter: bool = False


def decide_trade_side(
    p_up: float,
    market_up_price: float,
    market_down_price: float,
    edge_buffer: float = 0.02,
    regime: str = "unfiltered",
    allowed_side: str = "BOTH",
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

    if allowed_side == "NO_TRADE":
        return TradeDecision(
            side="NO_TRADE",
            p_side=0.0,
            market_price=0.0,
            edge=0.0,
            edge_up=edge_up,
            edge_down=edge_down,
            reason=f"Regime filter paused trading ({regime}).",
            regime=regime,
            allowed_side=allowed_side,
            blocked_by_regime=True,
            blocked_by_particle_filter=False,
        )

    if allowed_side == "UP":
        if edge_up <= edge_buffer:
            return TradeDecision(
                side="NO_TRADE",
                p_side=0.0,
                market_price=0.0,
                edge=edge_up,
                edge_up=edge_up,
                edge_down=edge_down,
                reason="Bull regime active, but UP edge is below threshold.",
                regime=regime,
                allowed_side=allowed_side,
                blocked_by_regime=False,
                blocked_by_particle_filter=False,
            )
        return TradeDecision(
            side="UP",
            p_side=p_up,
            market_price=market_up_price,
            edge=edge_up,
            edge_up=edge_up,
            edge_down=edge_down,
            reason="Bull regime allows only UP trades.",
            regime=regime,
            allowed_side=allowed_side,
            blocked_by_regime=False,
            blocked_by_particle_filter=False,
        )

    if allowed_side == "DOWN":
        if edge_down <= edge_buffer:
            return TradeDecision(
                side="NO_TRADE",
                p_side=0.0,
                market_price=0.0,
                edge=edge_down,
                edge_up=edge_up,
                edge_down=edge_down,
                reason="Bear regime active, but DOWN edge is below threshold.",
                regime=regime,
                allowed_side=allowed_side,
                blocked_by_regime=False,
                blocked_by_particle_filter=False,
            )
        return TradeDecision(
            side="DOWN",
            p_side=p_down,
            market_price=market_down_price,
            edge=edge_down,
            edge_up=edge_up,
            edge_down=edge_down,
            reason="Bear regime allows only DOWN trades.",
            regime=regime,
            allowed_side=allowed_side,
            blocked_by_regime=False,
            blocked_by_particle_filter=False,
        )

    if edge_up <= edge_buffer and edge_down <= edge_buffer:
        return TradeDecision(
            side="NO_TRADE",
            p_side=0.0,
            market_price=0.0,
            edge=max(edge_up, edge_down),
            edge_up=edge_up,
            edge_down=edge_down,
            reason="No edge above threshold",
            regime=regime,
            allowed_side=allowed_side,
            blocked_by_particle_filter=False,
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
            regime=regime,
            allowed_side=allowed_side,
            blocked_by_particle_filter=False,
        )

    return TradeDecision(
        side="DOWN",
        p_side=p_down,
        market_price=market_down_price,
        edge=edge_down,
        edge_up=edge_up,
        edge_down=edge_down,
        reason="DOWN has stronger model edge",
        regime=regime,
        allowed_side=allowed_side,
        blocked_by_particle_filter=False,
    )


def apply_particle_filter_entry_filter(
    decision: TradeDecision,
    observed_price: float,
    fair_price: float,
    min_gap: float = 0.0,
) -> TradeDecision:
    if decision.side not in {"UP", "DOWN"}:
        return decision

    if not (observed_price > 0 and fair_price > 0):
        return TradeDecision(
            side="NO_TRADE",
            p_side=0.0,
            market_price=0.0,
            edge=decision.edge,
            edge_up=decision.edge_up,
            edge_down=decision.edge_down,
            reason="Particle filter fair value was unavailable for entry confirmation.",
            regime=decision.regime,
            allowed_side=decision.allowed_side,
            blocked_by_regime=decision.blocked_by_regime,
            blocked_by_particle_filter=True,
        )

    if decision.side == "UP":
        if is_price_below_pf_fair_value(observed_price=observed_price, fair_price=fair_price, min_gap=min_gap):
            return decision
        return TradeDecision(
            side="NO_TRADE",
            p_side=0.0,
            market_price=0.0,
            edge=decision.edge,
            edge_up=decision.edge_up,
            edge_down=decision.edge_down,
            reason="Bull trade blocked because price is not below particle-filter fair value.",
            regime=decision.regime,
            allowed_side=decision.allowed_side,
            blocked_by_regime=decision.blocked_by_regime,
            blocked_by_particle_filter=True,
        )

    if is_price_above_pf_fair_value(observed_price=observed_price, fair_price=fair_price, min_gap=min_gap):
        return decision
    return TradeDecision(
        side="NO_TRADE",
        p_side=0.0,
        market_price=0.0,
        edge=decision.edge,
        edge_up=decision.edge_up,
        edge_down=decision.edge_down,
        reason="Bear trade blocked because price is not above particle-filter fair value.",
        regime=decision.regime,
        allowed_side=decision.allowed_side,
        blocked_by_regime=decision.blocked_by_regime,
        blocked_by_particle_filter=True,
    )
