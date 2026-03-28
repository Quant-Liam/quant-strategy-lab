from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kelly import KellyResult, kelly_fraction_binary


@dataclass(frozen=True)
class PFEdgeScoreResult:
    z_score: float
    raw_gap: float
    normalized_gap: float
    denom: float


@dataclass(frozen=True)
class PFProbabilityResult:
    p_base: float
    p_final: float
    confidence_multiplier: float


@dataclass(frozen=True)
class PFKellySizingResult:
    regime_label: str
    trade_side: str | None
    live_price: float
    fair_price_pf: float
    raw_gap: float
    normalized_gap: float
    z_score: float
    p_base: float | None
    p_final: float | None
    pf_confidence: float | None
    regime_confidence: float | None
    confidence_multiplier: float
    market_share_price: float | None
    kelly_fraction: float
    raw_kelly: float
    expected_log_growth: float
    break_even_prob: float | None
    effective_share_price: float | None
    net_odds: float | None
    no_trade_reason: str | None


def compute_pf_edge_score(
    live_price: float,
    fair_price_pf: float,
    pf_uncertainty: float,
    min_gap_scale: float = 0.001,
    epsilon: float = 1e-8,
    trade_side: str = "UP",
) -> PFEdgeScoreResult:
    """Normalize the PF fair-value gap into a dimensionless z-style edge score."""

    side = str(trade_side).upper()
    if side not in {"UP", "DOWN"}:
        raise ValueError("trade_side must be 'UP' or 'DOWN'")

    scale_floor = abs(float(live_price)) * max(float(min_gap_scale), 0.0)
    denom = float(max(abs(float(pf_uncertainty)), scale_floor, float(epsilon)))

    if side == "UP":
        raw_gap = float(fair_price_pf - live_price)
    else:
        raw_gap = float(live_price - fair_price_pf)

    normalized_gap = float(raw_gap / denom)
    return PFEdgeScoreResult(
        z_score=normalized_gap,
        raw_gap=raw_gap,
        normalized_gap=normalized_gap,
        denom=denom,
    )


def pf_gap_to_win_probability(
    z: float,
    alpha: float = 1.0,
    clip_min: float = 0.01,
    clip_max: float = 0.99,
    pf_confidence: float | None = None,
    regime_confidence: float | None = None,
    use_confidence_shrink: bool = True,
) -> PFProbabilityResult:
    """Map a normalized PF edge score into a clipped binary win probability."""

    sigmoid_input = float(np.clip(float(alpha) * float(z), -60.0, 60.0))
    p_base = float(1.0 / (1.0 + np.exp(-sigmoid_input)))
    p_base = float(np.clip(p_base, clip_min, clip_max))

    confidence_multiplier = 1.0
    if use_confidence_shrink:
        confidences = []
        for value in (pf_confidence, regime_confidence):
            if value is None:
                continue
            if np.isfinite(value):
                confidences.append(float(np.clip(value, 0.0, 1.0)))
        if confidences:
            confidence_multiplier = float(np.prod(confidences))

    p_final = 0.5 + (p_base - 0.5) * confidence_multiplier
    p_final = float(np.clip(p_final, clip_min, clip_max))

    return PFProbabilityResult(
        p_base=p_base,
        p_final=p_final,
        confidence_multiplier=confidence_multiplier,
    )


def compute_kelly_from_pf(
    live_price: float,
    fair_price_pf: float,
    pf_uncertainty: float,
    pf_confidence: float | None,
    regime_label: str,
    regime_confidence: float | None,
    market_share_price: float,
    fee_rate: float,
    alpha: float,
    min_gap_scale: float,
    fractional_kelly: float,
    max_fraction: float,
    epsilon: float = 1e-8,
    clip_min: float = 0.01,
    clip_max: float = 0.99,
    use_confidence_shrink: bool = True,
) -> PFKellySizingResult:
    """Convert PF fair value into a binary win probability, then into Kelly size.

    Example:
        A bull regime with BTC trading below PF fair value produces a positive
        z-score, which lifts p_win above 0.5 and can generate a non-zero Kelly
        fraction against the UP contract price.
    """

    regime = str(regime_label).lower()
    trade_side = {"bull": "UP", "bear": "DOWN"}.get(regime)

    result = _empty_result(
        regime_label=regime,
        trade_side=trade_side,
        live_price=live_price,
        fair_price_pf=fair_price_pf,
        pf_confidence=pf_confidence,
        regime_confidence=regime_confidence,
        market_share_price=market_share_price,
    )

    if trade_side is None:
        return _replace_reason(result, "neutral_regime")
    if not _is_finite_positive(live_price):
        return _replace_reason(result, "invalid_live_price")
    if not _is_finite_positive(fair_price_pf):
        return _replace_reason(result, "invalid_fair_price_pf")
    if not np.isfinite(pf_uncertainty):
        return _replace_reason(result, "invalid_pf_uncertainty")
    if not 0.0 < float(market_share_price) < 1.0:
        return _replace_reason(result, "invalid_market_share_price")

    edge = compute_pf_edge_score(
        live_price=live_price,
        fair_price_pf=fair_price_pf,
        pf_uncertainty=pf_uncertainty,
        min_gap_scale=min_gap_scale,
        epsilon=epsilon,
        trade_side=trade_side,
    )
    prob = pf_gap_to_win_probability(
        z=edge.z_score,
        alpha=alpha,
        clip_min=clip_min,
        clip_max=clip_max,
        pf_confidence=pf_confidence,
        regime_confidence=regime_confidence,
        use_confidence_shrink=use_confidence_shrink,
    )

    result = PFKellySizingResult(
        regime_label=regime,
        trade_side=trade_side,
        live_price=float(live_price),
        fair_price_pf=float(fair_price_pf),
        raw_gap=edge.raw_gap,
        normalized_gap=edge.normalized_gap,
        z_score=edge.z_score,
        p_base=prob.p_base,
        p_final=prob.p_final,
        pf_confidence=_optional_float(pf_confidence),
        regime_confidence=_optional_float(regime_confidence),
        confidence_multiplier=prob.confidence_multiplier,
        market_share_price=float(market_share_price),
        kelly_fraction=0.0,
        raw_kelly=0.0,
        expected_log_growth=0.0,
        break_even_prob=None,
        effective_share_price=None,
        net_odds=None,
        no_trade_reason=None,
    )

    if edge.raw_gap <= 0:
        return _replace_reason(result, "pf_gap_not_favorable")

    kelly = kelly_fraction_binary(
        p_win=prob.p_final,
        share_price=market_share_price,
        fee_rate=fee_rate,
        fractional_kelly=fractional_kelly,
        max_fraction=max_fraction,
    )

    result = PFKellySizingResult(
        regime_label=result.regime_label,
        trade_side=result.trade_side,
        live_price=result.live_price,
        fair_price_pf=result.fair_price_pf,
        raw_gap=result.raw_gap,
        normalized_gap=result.normalized_gap,
        z_score=result.z_score,
        p_base=result.p_base,
        p_final=result.p_final,
        pf_confidence=result.pf_confidence,
        regime_confidence=result.regime_confidence,
        confidence_multiplier=result.confidence_multiplier,
        market_share_price=result.market_share_price,
        kelly_fraction=kelly.fraction,
        raw_kelly=kelly.raw_kelly,
        expected_log_growth=kelly.expected_log_growth,
        break_even_prob=kelly.break_even_prob,
        effective_share_price=kelly.effective_share_price,
        net_odds=kelly.net_odds,
        no_trade_reason=None,
    )

    if prob.p_final <= kelly.break_even_prob:
        return _replace_reason(result, "p_win_below_break_even_after_fees")
    if kelly.fraction <= 0:
        return _replace_reason(result, "kelly_zero_after_risk_controls")

    return result


def _empty_result(
    regime_label: str,
    trade_side: str | None,
    live_price: float,
    fair_price_pf: float,
    pf_confidence: float | None,
    regime_confidence: float | None,
    market_share_price: float | None,
) -> PFKellySizingResult:
    return PFKellySizingResult(
        regime_label=str(regime_label),
        trade_side=trade_side,
        live_price=_optional_float(live_price, default=float("nan")),
        fair_price_pf=_optional_float(fair_price_pf, default=float("nan")),
        raw_gap=float("nan"),
        normalized_gap=float("nan"),
        z_score=float("nan"),
        p_base=None,
        p_final=None,
        pf_confidence=_optional_float(pf_confidence),
        regime_confidence=_optional_float(regime_confidence),
        confidence_multiplier=1.0,
        market_share_price=_optional_float(market_share_price),
        kelly_fraction=0.0,
        raw_kelly=0.0,
        expected_log_growth=0.0,
        break_even_prob=None,
        effective_share_price=None,
        net_odds=None,
        no_trade_reason=None,
    )


def _replace_reason(result: PFKellySizingResult, reason: str) -> PFKellySizingResult:
    return PFKellySizingResult(
        regime_label=result.regime_label,
        trade_side=result.trade_side,
        live_price=result.live_price,
        fair_price_pf=result.fair_price_pf,
        raw_gap=result.raw_gap,
        normalized_gap=result.normalized_gap,
        z_score=result.z_score,
        p_base=result.p_base,
        p_final=result.p_final,
        pf_confidence=result.pf_confidence,
        regime_confidence=result.regime_confidence,
        confidence_multiplier=result.confidence_multiplier,
        market_share_price=result.market_share_price,
        kelly_fraction=0.0,
        raw_kelly=0.0,
        expected_log_growth=0.0,
        break_even_prob=result.break_even_prob,
        effective_share_price=result.effective_share_price,
        net_odds=result.net_odds,
        no_trade_reason=reason,
    )


def _optional_float(value: float | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if np.isfinite(numeric):
        return numeric
    return default


def _is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and float(value) > 0.0)
