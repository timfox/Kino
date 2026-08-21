"""Congestion control performance metrics."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.cclab.perturbations import average_absolute_slope, log_scaled_cwnd_smoothness


def bandwidth_utilization(achieved_mbps: float, capacity_mbps: float) -> float:
    """Utilization U_t as percentage of available bandwidth."""
    if capacity_mbps <= 0:
        return 0.0
    return min(100.0, max(0.0, achieved_mbps / capacity_mbps * 100.0))


def cwnd_smoothness(cwnd: Tensor, window_k: int = 4) -> dict[str, float]:
    """Table IV: average absolute slope and log-scaled variant."""
    if cwnd.numel() < 2:
        return {"cwnd_smoothness": 0.0, "cwnd_smoothness_log": 0.0}
    diffs = (cwnd[1:] - cwnd[:-1]).abs()
    k = max(1, min(window_k, int(diffs.numel())))
    return {
        "cwnd_smoothness": float(average_absolute_slope(cwnd, k)),
        "cwnd_smoothness_log": log_scaled_cwnd_smoothness(cwnd, window_k=k),
    }


def max_utilization_degradation(
    clean: dict[str, float],
    adversarial: dict[str, float],
) -> float:
    """Worst-case utilization drop (%) under delay constraint."""
    return max(0.0, clean.get("util_pct", 0.0) - adversarial.get("util_pct", 0.0))
