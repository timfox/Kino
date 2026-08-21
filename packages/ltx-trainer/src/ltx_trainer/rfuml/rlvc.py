"""RLVC training stages (Sec. III-D, Eq. 12–15)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def cyclical_learning_rate(
    epoch: int,
    *,
    eta_max: float,
    eta_min: float,
    cycle_length: int,
) -> float:
    r"""Linear cyclical schedule (Eq. 12). ``epoch`` is 1-indexed."""
    if cycle_length <= 1:
        return eta_min
    t = epoch - 1
    s = (t % cycle_length) / (cycle_length - 1)
    return s * eta_min + (1.0 - s) * eta_max


@dataclass
class GMMPartition:
    """Two-component GMM partition result for one view's average losses."""

    clean_indices: list[int]
    conflicting_indices: list[int]
    posterior_clean: list[float]


def _log_gaussian(x: float, mean: float, var: float) -> float:
    var = max(var, 1e-6)
    return -0.5 * math.log(2.0 * math.pi * var) - 0.5 * ((x - mean) ** 2) / var


def fit_gmm_two_component(losses: Sequence[float], *, max_iter: int = 50) -> GMMPartition:
    r"""EM fit for clean vs conflicting partition (Stage 3); simplified 1D GMM."""
    xs = [float(x) for x in losses]
    n = len(xs)
    if n == 0:
        return GMMPartition([], [], [])
    # Init: low-mean = clean, high-mean = conflicting
    sorted_idx = sorted(range(n), key=lambda i: xs[i])
    mid = n // 2
    mu0 = sum(xs[i] for i in sorted_idx[:mid]) / max(mid, 1)
    mu1 = sum(xs[i] for i in sorted_idx[mid:]) / max(n - mid, 1)
    var0 = var1 = max((sum(x * x for x in xs) / n) - (sum(xs) / n) ** 2, 1e-4)
    pi0 = pi1 = 0.5
    resp_clean = [0.5] * n
    for _ in range(max_iter):
        # E-step
        for i, x in enumerate(xs):
            p0 = pi0 * math.exp(_log_gaussian(x, mu0, var0))
            p1 = pi1 * math.exp(_log_gaussian(x, mu1, var1))
            denom = p0 + p1 + 1e-12
            resp_clean[i] = p0 / denom
        # M-step
        n0 = sum(resp_clean)
        n1 = n - n0
        if n0 < 1e-6 or n1 < 1e-6:
            break
        pi0 = n0 / n
        pi1 = n1 / n
        mu0 = sum(resp_clean[i] * xs[i] for i in range(n)) / n0
        mu1 = sum((1.0 - resp_clean[i]) * xs[i] for i in range(n)) / n1
        var0 = sum(resp_clean[i] * (xs[i] - mu0) ** 2 for i in range(n)) / n0 + 1e-6
        var1 = sum((1.0 - resp_clean[i]) * (xs[i] - mu1) ** 2 for i in range(n)) / n1 + 1e-6
    clean_idx = [i for i in range(n) if resp_clean[i] > 0.5]
    con_idx = [i for i in range(n) if resp_clean[i] <= 0.5]
    return GMMPartition(clean_idx, con_idx, resp_clean)


def importance_weight(is_clean: bool, posterior_clean: float) -> float:
    r"""``d_i`` in Eq. 15: 1 if clean else ``p(k=0|L)``."""
    return 1.0 if is_clean else posterior_clean


def per_view_average_loss(loss_trajectory: Sequence[float]) -> float:
    r"""``L^v_i`` averaged over cyclical epochs (Eq. 13)."""
    if not loss_trajectory:
        return 0.0
    return sum(loss_trajectory) / len(loss_trajectory)
