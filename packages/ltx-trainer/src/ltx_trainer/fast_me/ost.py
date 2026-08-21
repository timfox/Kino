"""Optimal Stopping Theory rules for adaptive motion estimation (Sec. III)."""

from __future__ import annotations

import math
from typing import Sequence


def empirical_cdf(samples: Sequence[float], y: float) -> float:
    r"""F_Y(y) = (1/N) sum I(Y_i <= y) (Eq. 3)."""
    if not samples:
        return 0.0
    return sum(1 for s in samples if s <= y) / len(samples)


def exponential_cdf(y: float, *, theta: float) -> float:
    r"""F_Y(y) = 1 - exp(-θ y) for y >= 0 (Eq. 8)."""
    if y < 0:
        return 0.0
    return 1.0 - math.exp(-theta * y)


def ost_stop_index(
    sad_values: Sequence[float],
    *,
    delta: float = 0.05,
) -> int | None:
    r"""τ* = min{k | F_Y(Y_k) >= 1 - δ} using empirical CDF (Eq. 4)."""
    seen: list[float] = []
    for k, yk in enumerate(sad_values, start=1):
        seen.append(yk)
        if empirical_cdf(seen, yk) >= 1.0 - delta:
            return k
    return None


def exponential_stop_threshold(*, delta: float, theta: float) -> float:
    r"""Y_k >= -log(δ) / θ (Eq. 5)."""
    if delta <= 0 or theta <= 0:
        raise ValueError("delta and theta must be positive")
    return -math.log(delta) / theta


def exponential_tau_star(*, delta: float, theta: float) -> float:
    r"""Closed-form τ* = (1 - e^{-θ τ*}) / θ under exponential SAD model (Eq. 27)."""
    if theta <= 0:
        raise ValueError("theta must be positive")
    # Fixed point: τ = (1 - exp(-θ τ)) / θ  →  solve numerically once.
    tau = 1.0 / theta
    for _ in range(32):
        tau = (1.0 - math.exp(-theta * tau)) / theta
    return tau
