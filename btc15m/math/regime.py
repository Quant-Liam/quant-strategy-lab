from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


RegimeState = str
REGIME_TO_SIDE: dict[RegimeState, str] = {
    "bull": "UP",
    "neutral": "NO_TRADE",
    "bear": "DOWN",
}


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: RegimeState
    allowed_side: str
    confidence: float
    probabilities: dict[RegimeState, float]
    state_mean_returns: dict[RegimeState, float]
    state_volatility: dict[RegimeState, float]
    garch_volatility: float
    garch_volatility_ratio: float
    fit_status: str
    reason: str


def fit_markov_garch_regime(
    candles_15m: pd.DataFrame,
    lookback: int = 240,
    min_history: int = 80,
) -> RegimeSnapshot:
    regime_frame = compute_regime_frame(
        candles_15m=candles_15m,
        lookback=lookback,
        min_history=min_history,
    )
    return snapshot_from_regime_frame(regime_frame)


def snapshot_from_regime_frame(regime_frame: pd.DataFrame) -> RegimeSnapshot:
    if regime_frame.empty:
        return _fallback_snapshot(pd.Series(dtype=float), fit_status="insufficient_history")

    last = regime_frame.iloc[-1]
    probabilities = {
        "bull": float(last.get("bull_prob", 0.0)),
        "neutral": float(last.get("neutral_prob", 0.0)),
        "bear": float(last.get("bear_prob", 0.0)),
    }
    state_mean_returns = {
        "bull": float(last.get("bull_mean_return", 0.0)),
        "neutral": float(last.get("neutral_mean_return", 0.0)),
        "bear": float(last.get("bear_mean_return", 0.0)),
    }
    state_volatility = {
        "bull": float(last.get("bull_volatility", 0.0)),
        "neutral": float(last.get("neutral_volatility", 0.0)),
        "bear": float(last.get("bear_volatility", 0.0)),
    }
    regime = str(last.get("regime", "neutral"))
    confidence = float(last.get("confidence", 0.0))
    return RegimeSnapshot(
        regime=regime,
        allowed_side=REGIME_TO_SIDE.get(regime, "NO_TRADE"),
        confidence=confidence,
        probabilities=probabilities,
        state_mean_returns=state_mean_returns,
        state_volatility=state_volatility,
        garch_volatility=float(last.get("garch_volatility", 0.0)),
        garch_volatility_ratio=float(last.get("garch_volatility_ratio", 1.0)),
        fit_status=str(last.get("fit_status", "fit")),
        reason=str(last.get("reason", "")),
    )


def compute_regime_frame(
    candles_15m: pd.DataFrame,
    lookback: int = 240,
    min_history: int = 80,
) -> pd.DataFrame:
    closes = _extract_close_series(candles_15m)
    if closes.empty:
        return pd.DataFrame()

    if lookback > 0:
        closes = closes.tail(max(lookback, min_history + 1))

    # 15m log return:
    # r_t = log(P_t) - log(P_{t-1})
    returns = np.log(closes).diff().dropna()
    if len(returns) < min_history:
        snapshot = _fallback_snapshot(returns, fit_status="insufficient_history")
        return _snapshot_to_frame(returns.index, snapshot)

    try:
        # First estimate a simple GARCH(1,1) volatility filter so regime
        # fitting sees returns normalized by conditional risk.
        garch_variance = _fit_garch11_variance(returns)
        # Standardized return:
        # z_t = r_t / sigma_t
        standardized_returns = (returns / np.sqrt(garch_variance)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(standardized_returns) < min_history:
            snapshot = _fallback_snapshot(returns, fit_status="garch_standardization_failed")
            return _snapshot_to_frame(returns.index, snapshot)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            # statsmodels then fits a 3-state Markov switching regression with
            # switching variance on z_t. The hidden state is the regime.
            model = MarkovRegression(
                standardized_returns,
                k_regimes=3,
                trend="c",
                switching_variance=True,
            )
            result = model.fit(disp=False, maxiter=100, em_iter=5)

        probabilities = result.filtered_marginal_probabilities
        if not isinstance(probabilities, pd.DataFrame):
            probabilities = pd.DataFrame(
                probabilities,
                index=standardized_returns.index,
                columns=[0, 1, 2],
            )

        mapping = _label_regimes(returns.loc[standardized_returns.index], probabilities, garch_variance.loc[standardized_returns.index])
        labeled = pd.DataFrame(index=standardized_returns.index)
        mean_map = {}
        vol_map = {}
        for raw_state, label in mapping.items():
            mean_return, vol_value = _weighted_state_stats(
                returns.loc[standardized_returns.index],
                garch_variance.loc[standardized_returns.index],
                probabilities[raw_state],
            )
            labeled[f"{label}_prob"] = probabilities[raw_state].astype(float)
            mean_map[label] = mean_return
            vol_map[label] = vol_value

        for label in REGIME_TO_SIDE:
            if f"{label}_prob" not in labeled:
                labeled[f"{label}_prob"] = 0.0
            labeled[f"{label}_mean_return"] = float(mean_map.get(label, 0.0))
            labeled[f"{label}_volatility"] = float(vol_map.get(label, 0.0))

        ordered = ["bull", "neutral", "bear"]
        regime_prob_frame = labeled[[f"{name}_prob" for name in ordered]]
        labeled["regime"] = regime_prob_frame.idxmax(axis=1).str.replace("_prob", "", regex=False)
        labeled["confidence"] = regime_prob_frame.max(axis=1).astype(float)
        labeled["garch_volatility"] = np.sqrt(garch_variance.loc[standardized_returns.index]).astype(float)
        rolling_baseline = labeled["garch_volatility"].rolling(32, min_periods=8).mean()
        baseline_fallback = float(labeled["garch_volatility"].mean()) if not labeled.empty else 1.0
        rolling_baseline = rolling_baseline.fillna(baseline_fallback).replace(0.0, np.nan)
        labeled["garch_volatility_ratio"] = (
            labeled["garch_volatility"] / rolling_baseline.fillna(baseline_fallback)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # These are hand-tuned guardrails, not model-estimated parameters.
        # We demote weak or sign-inconsistent states to neutral.
        low_conf_mask = labeled["confidence"] < 0.45
        weak_drift_mask = labeled["regime"].eq("bull") & (labeled["bull_mean_return"] <= 0)
        weak_drift_mask |= labeled["regime"].eq("bear") & (labeled["bear_mean_return"] >= 0)
        labeled.loc[low_conf_mask | weak_drift_mask, "regime"] = "neutral"
        labeled["allowed_side"] = labeled["regime"].map(REGIME_TO_SIDE).fillna("NO_TRADE")
        labeled["fit_status"] = "fit"
        labeled["reason"] = labeled.apply(_build_reason, axis=1)
        return labeled
    except Exception:
        snapshot = _fallback_snapshot(returns, fit_status="fallback")
        return _snapshot_to_frame(returns.index, snapshot)


def _extract_close_series(candles_15m: pd.DataFrame) -> pd.Series:
    if candles_15m.empty or "close" not in candles_15m.columns:
        return pd.Series(dtype=float)
    closes = candles_15m["close"].astype(float).copy()
    closes = closes.dropna()
    if not isinstance(closes.index, pd.DatetimeIndex):
        closes.index = pd.to_datetime(closes.index, utc=True)
    elif closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC")
    else:
        closes.index = closes.index.tz_convert("UTC")
    return closes.sort_index()


def _fit_garch11_variance(returns: pd.Series) -> pd.Series:
    values = returns.to_numpy(dtype=float)
    initial_variance = float(max(np.var(values), 1e-8))

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        omega = float(np.exp(theta[0]))
        alpha_raw = float(np.exp(theta[1]))
        beta_raw = float(np.exp(theta[2]))
        scale = 1.0 + alpha_raw + beta_raw
        alpha = 0.995 * alpha_raw / scale
        beta = 0.995 * beta_raw / scale
        return omega, alpha, beta

    def neg_log_likelihood(theta: np.ndarray) -> float:
        omega, alpha, beta = unpack(theta)
        variance = np.empty_like(values)
        variance[0] = initial_variance
        for i in range(1, len(values)):
            # GARCH(1,1):
            # sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
            variance[i] = omega + alpha * (values[i - 1] ** 2) + beta * variance[i - 1]
            variance[i] = max(variance[i], 1e-10)
        # Gaussian quasi-log-likelihood up to constants.
        ll = 0.5 * np.sum(np.log(variance) + (values**2) / variance)
        return float(ll)

    # The initial guess biases beta high and alpha modest because BTC volatility
    # usually clusters more through persistence than single-shock reaction.
    start = np.log(np.array([initial_variance * 0.05, 0.08, 0.90], dtype=float))
    opt = minimize(neg_log_likelihood, start, method="L-BFGS-B")
    omega, alpha, beta = unpack(opt.x if opt.success else start)

    variance = np.empty_like(values)
    variance[0] = initial_variance
    for i in range(1, len(values)):
        variance[i] = omega + alpha * (values[i - 1] ** 2) + beta * variance[i - 1]
        variance[i] = max(variance[i], 1e-10)

    return pd.Series(variance, index=returns.index, name="garch_variance")


def _label_regimes(
    returns: pd.Series,
    probabilities: pd.DataFrame,
    garch_variance: pd.Series,
) -> dict[int, RegimeState]:
    # Regime labels are assigned after fitting by ordering states from the most
    # negative weighted mean return to the most positive.
    state_scores: list[tuple[int, float, float]] = []
    for state in probabilities.columns:
        mean_return, vol_value = _weighted_state_stats(returns, garch_variance, probabilities[state])
        state_scores.append((int(state), mean_return, vol_value))

    state_scores.sort(key=lambda row: (row[1], -row[2]))
    if len(state_scores) != 3:
        raise ValueError("Expected exactly three regimes for bull/neutral/bear mapping")

    return {
        state_scores[0][0]: "bear",
        state_scores[1][0]: "neutral",
        state_scores[2][0]: "bull",
    }


def _weighted_state_stats(
    returns: pd.Series,
    garch_variance: pd.Series,
    weights: pd.Series,
) -> tuple[float, float]:
    # Weighted mean and volatility under each regime probability path.
    aligned_returns = returns.loc[weights.index].to_numpy(dtype=float)
    aligned_variance = garch_variance.loc[weights.index].to_numpy(dtype=float)
    aligned_weights = weights.to_numpy(dtype=float)
    total_weight = float(aligned_weights.sum())
    if total_weight <= 1e-12:
        return 0.0, float(np.sqrt(np.nanmean(aligned_variance))) if aligned_variance.size else 0.0
    mean_return = float(np.dot(aligned_returns, aligned_weights) / total_weight)
    avg_volatility = float(np.dot(np.sqrt(aligned_variance), aligned_weights) / total_weight)
    return mean_return, avg_volatility


def _build_reason(row: pd.Series) -> str:
    regime = str(row.get("regime", "neutral"))
    confidence = float(row.get("confidence", 0.0))
    drift = float(row.get(f"{regime}_mean_return", 0.0))
    vol_ratio = float(row.get("garch_volatility_ratio", 1.0))
    if regime == "bull":
        return (
            f"Bull regime with {confidence * 100:.1f}% confidence, "
            f"positive drift {drift * 100:.3f}% per 15m and vol ratio {vol_ratio:.2f}."
        )
    if regime == "bear":
        return (
            f"Bear regime with {confidence * 100:.1f}% confidence, "
            f"negative drift {drift * 100:.3f}% per 15m and vol ratio {vol_ratio:.2f}."
        )
    return (
        f"Neutral regime with {confidence * 100:.1f}% confidence. "
        f"Signal drift is mixed, so trading is paused."
    )


def _fallback_snapshot(returns: pd.Series, fit_status: str) -> RegimeSnapshot:
    # Fallback is deliberately simple and heuristic; it keeps the dashboard
    # usable if the Markov-GARCH optimization fails.
    short = float(returns.tail(8).mean()) if not returns.empty else 0.0
    medium = float(returns.tail(24).mean()) if len(returns) >= 8 else short
    vol = float(returns.tail(32).std()) if len(returns) >= 8 else 0.0
    baseline = float(returns.std()) if len(returns) >= 8 else max(vol, 1e-8)
    vol_ratio = float(vol / baseline) if baseline > 1e-8 else 1.0

    if medium > 0.0004 and short >= -0.0002:
        regime = "bull"
        probabilities = {"bull": 0.60, "neutral": 0.25, "bear": 0.15}
    elif medium < -0.0004 and short <= 0.0002:
        regime = "bear"
        probabilities = {"bull": 0.15, "neutral": 0.25, "bear": 0.60}
    else:
        regime = "neutral"
        probabilities = {"bull": 0.20, "neutral": 0.60, "bear": 0.20}

    state_mean_returns = {"bull": max(medium, 0.0), "neutral": 0.0, "bear": min(medium, 0.0)}
    state_volatility = {"bull": vol, "neutral": vol, "bear": vol}
    reason = (
        "Fallback regime classifier is active because the Markov-GARCH fit "
        f"was unavailable ({fit_status})."
    )
    return RegimeSnapshot(
        regime=regime,
        allowed_side=REGIME_TO_SIDE[regime],
        confidence=float(max(probabilities.values())),
        probabilities=probabilities,
        state_mean_returns=state_mean_returns,
        state_volatility=state_volatility,
        garch_volatility=vol,
        garch_volatility_ratio=vol_ratio if np.isfinite(vol_ratio) else 1.0,
        fit_status=fit_status,
        reason=reason,
    )


def _snapshot_to_frame(index: pd.Index, snapshot: RegimeSnapshot) -> pd.DataFrame:
    if len(index) == 0:
        return pd.DataFrame()
    frame = pd.DataFrame(index=index)
    for label, value in snapshot.probabilities.items():
        frame[f"{label}_prob"] = float(value)
    for label, value in snapshot.state_mean_returns.items():
        frame[f"{label}_mean_return"] = float(value)
    for label, value in snapshot.state_volatility.items():
        frame[f"{label}_volatility"] = float(value)
    frame["regime"] = snapshot.regime
    frame["allowed_side"] = snapshot.allowed_side
    frame["confidence"] = snapshot.confidence
    frame["garch_volatility"] = snapshot.garch_volatility
    frame["garch_volatility_ratio"] = snapshot.garch_volatility_ratio
    frame["fit_status"] = snapshot.fit_status
    frame["reason"] = snapshot.reason
    return frame
