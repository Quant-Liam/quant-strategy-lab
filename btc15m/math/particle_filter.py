from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regime import RegimeState, compute_regime_frame


@dataclass(frozen=True)
class ParticleFilterConfig:
    num_particles: int = 300
    resample_threshold: float = 0.50
    lookback: int = 240


@dataclass(frozen=True)
class ParticleFilterSnapshot:
    observed_price: float
    fair_price: float
    gap: float
    gap_pct: float
    drift: float
    uncertainty: float
    confidence: float
    regime: RegimeState
    price_below_fair: bool
    price_above_fair: bool
    bull_long_setup: bool
    bear_short_setup: bool


def fit_regime_aware_particle_filter(
    candles_15m: pd.DataFrame,
    regime_frame: pd.DataFrame | None = None,
    config: ParticleFilterConfig | None = None,
) -> ParticleFilterSnapshot:
    pf_frame = compute_particle_filter_frame(
        candles_15m=candles_15m,
        regime_frame=regime_frame,
        config=config,
    )
    return snapshot_from_particle_filter_frame(pf_frame)


def compute_particle_filter_frame(
    candles_15m: pd.DataFrame,
    regime_frame: pd.DataFrame | None = None,
    config: ParticleFilterConfig | None = None,
) -> pd.DataFrame:
    active_config = config or ParticleFilterConfig()
    closes = _extract_close_series(candles_15m)
    if closes.empty:
        return pd.DataFrame()

    if active_config.lookback > 0:
        closes = closes.tail(active_config.lookback)

    if regime_frame is None or regime_frame.empty:
        regime_frame = compute_regime_frame(
            candles_15m=candles_15m,
            lookback=active_config.lookback,
        )

    # Observation is handled in log-price space so drift and noise behave like
    # percentage moves rather than raw BTC-dollar moves.
    log_prices = np.log(closes.to_numpy(dtype=float))
    n_particles = int(max(active_config.num_particles, 50))
    rng = np.random.default_rng(42)

    initial_regime_row = _lookup_regime_row(regime_frame, closes.index[0])
    initial_scale = _regime_scale(initial_regime_row)
    fair_particles = log_prices[0] + rng.normal(scale=max(initial_scale["obs_noise"], 1e-4), size=n_particles)
    drift_particles = np.full(n_particles, initial_scale["drift_anchor"], dtype=float)
    drift_particles += rng.normal(scale=max(initial_scale["drift_noise"], 1e-5), size=n_particles)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)

    rows: list[dict[str, object]] = []

    # Latent state:
    # f_t = latent fair log-price
    # d_t = latent drift
    for ts, observed_log_price, observed_price in zip(closes.index, log_prices, closes.to_numpy(dtype=float), strict=True):
        regime_row = _lookup_regime_row(regime_frame, ts)
        scale = _regime_scale(regime_row)
        regime = str(regime_row.get("regime", "neutral"))

        # Regime-conditioned transition:
        # d_t = rho_r d_{t-1} + (1 - rho_r) mu_r + eta_t
        drift_particles = (
            scale["drift_persistence"] * drift_particles
            + (1.0 - scale["drift_persistence"]) * scale["drift_anchor"]
            + rng.normal(scale=scale["drift_noise"], size=n_particles)
        )
        # Fair value transition:
        # f_t = f_{t-1} + d_t + epsilon_t
        # rng. is a random number generator that produces the same sequence of numbers every time for reproducibility. The noise terms are normally distributed with standard deviations defined by the current regime's scales.
        fair_particles = fair_particles + drift_particles + rng.normal(scale=scale["fair_noise"], size=n_particles)

        # Observation model:
        # y_t = f_t + nu_t,  nu_t ~ N(0, obs_noise^2)
        log_likelihood = (
            -0.5 * ((observed_log_price - fair_particles) / scale["obs_noise"]) ** 2
            - np.log(scale["obs_noise"])
        )
        weights = _normalize_log_weights(np.log(weights + 1e-300) + log_likelihood)

        # Effective sample size:
        # ESS = 1 / sum_i w_i^2
        ess = 1.0 / np.sum(weights**2)
        fair_log_estimate = float(np.dot(weights, fair_particles))
        drift_estimate = float(np.dot(weights, drift_particles))
        fair_log_var = float(np.dot(weights, (fair_particles - fair_log_estimate) ** 2))

        fair_price = float(np.exp(fair_log_estimate))
        uncertainty = float(fair_price * np.sqrt(max(fair_log_var, 0.0)))
        gap = get_pf_gap(observed_price=float(observed_price), fair_price=fair_price)
        gap_pct = float(gap / fair_price) if fair_price > 0 else float("nan")
        confidence = float(ess / n_particles)

        price_below_fair = is_price_below_pf_fair_value(float(observed_price), fair_price)
        price_above_fair = is_price_above_pf_fair_value(float(observed_price), fair_price)

        rows.append(
            {
                "price": float(observed_price),
                "regime": regime,
                "pf_fair_price": fair_price,
                "pf_gap": gap,
                "pf_gap_pct": gap_pct,
                "pf_drift": drift_estimate,
                "pf_uncertainty": uncertainty,
                "pf_confidence": confidence,
                "pf_price_below_fair": price_below_fair,
                "pf_price_above_fair": price_above_fair,
                "pf_bull_long_setup": regime == "bull" and price_below_fair,
                "pf_bear_short_setup": regime == "bear" and price_above_fair,
            }
        )

        # Resample once the weight concentration gets too high.
        if ess < n_particles * active_config.resample_threshold:
            fair_particles, drift_particles = _systematic_resample(
                fair_particles=fair_particles,
                drift_particles=drift_particles,
                weights=weights,
                rng=rng,
            )
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)

    return pd.DataFrame(rows, index=closes.index)


def get_pf_gap(observed_price: float, fair_price: float) -> float:
    return float(observed_price - fair_price)


def is_price_below_pf_fair_value(observed_price: float, fair_price: float, min_gap: float = 0.0) -> bool:
    return bool(get_pf_gap(observed_price, fair_price) < -abs(min_gap))


def is_price_above_pf_fair_value(observed_price: float, fair_price: float, min_gap: float = 0.0) -> bool:
    return bool(get_pf_gap(observed_price, fair_price) > abs(min_gap))


def project_particle_filter_fair_price(
    fair_price: float,
    drift: float,
    step_fraction: float = 1.0,
) -> float:
    clipped_fraction = float(np.clip(step_fraction, 0.0, 1.5))
    return float(fair_price * np.exp(drift * clipped_fraction))


def project_particle_filter_to_time(
    snapshot: ParticleFilterSnapshot,
    last_observation_time: pd.Timestamp,
    target_time: pd.Timestamp,
    step: pd.Timedelta = pd.Timedelta(minutes=15),
) -> float:
    if step <= pd.Timedelta(0):
        raise ValueError("step must be positive")
    elapsed = (target_time - last_observation_time).total_seconds()
    fraction = elapsed / step.total_seconds()
    return project_particle_filter_fair_price(
        fair_price=snapshot.fair_price,
        drift=snapshot.drift,
        step_fraction=fraction,
    )


def snapshot_from_particle_filter_frame(pf_frame: pd.DataFrame) -> ParticleFilterSnapshot:
    if pf_frame.empty:
        return ParticleFilterSnapshot(
            observed_price=float("nan"),
            fair_price=float("nan"),
            gap=float("nan"),
            gap_pct=float("nan"),
            drift=0.0,
            uncertainty=float("nan"),
            confidence=0.0,
            regime="neutral",
            price_below_fair=False,
            price_above_fair=False,
            bull_long_setup=False,
            bear_short_setup=False,
        )

    last = pf_frame.iloc[-1]
    return ParticleFilterSnapshot(
        observed_price=float(last["price"]),
        fair_price=float(last["pf_fair_price"]),
        gap=float(last["pf_gap"]),
        gap_pct=float(last["pf_gap_pct"]),
        drift=float(last["pf_drift"]),
        uncertainty=float(last["pf_uncertainty"]),
        confidence=float(last["pf_confidence"]),
        regime=str(last["regime"]),
        price_below_fair=bool(last["pf_price_below_fair"]),
        price_above_fair=bool(last["pf_price_above_fair"]),
        bull_long_setup=bool(last["pf_bull_long_setup"]),
        bear_short_setup=bool(last["pf_bear_short_setup"]),
    )


def _extract_close_series(candles_15m: pd.DataFrame) -> pd.Series:
    if candles_15m.empty or "close" not in candles_15m.columns:
        return pd.Series(dtype=float)
    closes = candles_15m["close"].astype(float).copy().dropna()
    if not isinstance(closes.index, pd.DatetimeIndex):
        closes.index = pd.to_datetime(closes.index, utc=True)
    elif closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC")
    else:
        closes.index = closes.index.tz_convert("UTC")
    return closes.sort_index()


def _lookup_regime_row(regime_frame: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float | str]:
    if regime_frame.empty:
        return {"regime": "neutral"}
    if ts in regime_frame.index:
        return regime_frame.loc[ts].to_dict()

    subset = regime_frame.loc[regime_frame.index <= ts]
    if not subset.empty:
        return subset.iloc[-1].to_dict()
    return regime_frame.iloc[0].to_dict()


def _regime_scale(regime_row: dict[str, float | str]) -> dict[str, float]:
    # The MS-GARCH regime does not get replaced here; it only sets PF dynamics.
    # drift_anchor and volatility scales are derived from the current regime's
    # estimated drift and conditional volatility.
    regime = str(regime_row.get("regime", "neutral"))
    regime_vol = abs(float(regime_row.get("garch_volatility", 0.0015)))
    bull_anchor = max(float(regime_row.get("bull_mean_return", 0.0005)), 0.0001)
    bear_anchor = min(float(regime_row.get("bear_mean_return", -0.0005)), -0.0001)
    neutral_anchor = float(regime_row.get("neutral_mean_return", 0.0))

    if regime == "bull":
        drift_anchor = bull_anchor
        drift_persistence = 0.96
        fair_noise = max(0.0004, regime_vol * 0.40)
    elif regime == "bear":
        drift_anchor = bear_anchor
        drift_persistence = 0.96
        fair_noise = max(0.0004, regime_vol * 0.40)
    else:
        drift_anchor = neutral_anchor
        drift_persistence = 0.90
        fair_noise = max(0.0005, regime_vol * 0.55)

    return {
        "drift_anchor": drift_anchor,
        "drift_persistence": drift_persistence,
        "drift_noise": max(0.00005, regime_vol * 0.10),
        "fair_noise": fair_noise,
        "obs_noise": max(0.0008, regime_vol * 0.65),
    }


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _systematic_resample(
    fair_particles: np.ndarray,
    drift_particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_particles = len(weights)
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    cumulative = np.cumsum(weights)
    indexes = np.searchsorted(cumulative, positions, side="left")
    return fair_particles[indexes].copy(), drift_particles[indexes].copy()
