"""Drift-aware and uncertainty-aware autoscaling (Sec. VII)."""

from __future__ import annotations

import math
from typing import Any


def autoscaling_drift_index(
    observed: list[float],
    predicted: list[float],
) -> float:
    """ADI = mean |R_t - R̂_t| over interval (Eq. 15)."""
    if not observed or len(observed) != len(predicted):
        return 0.0
    return sum(abs(o - p) for o, p in zip(observed, predicted, strict=True)) / len(observed)


def needs_drift_correction(adi: float, tau_adi: float) -> bool:
    return adi > tau_adi


def uncertainty_correction(
    s_pred: float,
    *,
    mean_error: float,
    alpha: float = 0.5,
    kappa: float = 0.3,
    s_max: float = 50.0,
) -> dict[str, float]:
    """Bounded feedback correction (Eq. 17–19)."""
    delta = alpha * mean_error
    delta = max(-kappa * s_max, min(kappa * s_max, delta))
    return {"s_pred": s_pred, "delta_s": delta, "s_corrected": s_pred + delta}


def resource_removal_strategy(
    current_replicas: int,
    target_replicas: int,
    *,
    gamma: float = 0.6,
) -> int:
    """RRS gradual scale-in (Eq. 20)."""
    if target_replicas >= current_replicas:
        return current_replicas
    remove = gamma * (current_replicas - target_replicas)
    return max(target_replicas, int(current_replicas - remove))


def frsc_straggler_ids(
    predicted_train_times: dict[str, float],
    *,
    epsilon: float = 0.2,
) -> list[str]:
    """Mark clients with τ̂_i > (1+ε) τ_median (Eq. 21)."""
    if not predicted_train_times:
        return []
    times = sorted(predicted_train_times.values())
    median = times[len(times) // 2]
    threshold = (1.0 + epsilon) * median
    return [cid for cid, t in predicted_train_times.items() if t > threshold]


def frsc_boost(
    predicted_time: float,
    median_time: float,
    *,
    beta: float = 1.0,
) -> float:
    """Targeted corrective boost ΔS_i (Eq. 23)."""
    return max(0.0, beta * (predicted_time - median_time))


def drift_demo(
    observed_cpu: list[float],
    predicted_cpu: list[float],
    *,
    tau_adi: float = 0.15,
) -> dict[str, Any]:
    adi = autoscaling_drift_index(observed_cpu, predicted_cpu)
    err = [o - p for o, p in zip(observed_cpu, predicted_cpu, strict=True)]
    mean_err = sum(err) / len(err) if err else 0.0
    corr = uncertainty_correction(8.0, mean_error=mean_err)
    return {
        "adi": round(adi, 4),
        "tau_adi": tau_adi,
        "correct": needs_drift_correction(adi, tau_adi),
        "correction": corr,
    }
