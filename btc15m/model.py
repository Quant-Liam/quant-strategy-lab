from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalWeights:
    bias: float = 0.0
    mom_30m: float = 0.55
    mom_60m: float = 0.75
    mom_120m: float = 0.35
    vol_2h: float = -0.25
    stalling: float = -0.45
    live_vs_lastclose: float = 0.65
    depth_imbalance: float = 0.85
    wall_skew_log: float = 0.55
    top_wall_distance_skew: float = -0.20
    rsi_centered: float = 0.20
    funding_zscore: float = -0.40
    liquidation_imbalance: float = -0.35
    news_shock: float = 0.50


@dataclass(frozen=True)
class ModelOutput:
    p_up: float
    score: float
    normalized_features: dict[str, float]
    contributions: dict[str, float]


FEATURE_SCALES: dict[str, float] = {
    "mom_30m": 350.0,
    "mom_60m": 280.0,
    "mom_120m": 200.0,
    "vol_2h": 250.0,
    "live_vs_lastclose": 450.0,
    "depth_imbalance": 3.0,
    "wall_skew_log": 1.0,
    "top_wall_distance_skew": 0.2,
    "rsi_centered": 2.0,
    "stalling": 1.0,
    "funding_zscore": 1.0,
    "liquidation_imbalance": 2.0,
    "news_shock": 2.0,
}


def _sigmoid(x: float) -> float:
    clipped = float(np.clip(x, -60, 60))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def normalize_features(raw_features: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, value in raw_features.items():
        scale = FEATURE_SCALES.get(name, 1.0)
        normalized[name] = float(np.clip(value * scale, -6.0, 6.0))
    return normalized


def predict_up_probability(
    features: dict[str, float],
    weights: SignalWeights | None = None,
) -> ModelOutput:
    active_weights = weights or SignalWeights()
    normalized = normalize_features(features)

    score = active_weights.bias
    contributions: dict[str, float] = {}

    for name, value in normalized.items():
        weight = getattr(active_weights, name, None)
        if weight is None:
            continue
        contrib = float(weight * value)
        contributions[name] = contrib
        score += contrib

    p_up = _sigmoid(score)
    return ModelOutput(
        p_up=p_up,
        score=float(score),
        normalized_features=normalized,
        contributions=contributions,
    )
