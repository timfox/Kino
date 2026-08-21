"""Regret bound — Theorem 1, Eq. (32)."""

from __future__ import annotations

import math


def universal_regret_bound(
    k_segments: int,
    penalty_c: float,
    tau: int,
    l_max: float,
    dim_p: int,
    horizon_t: int,
    gamma: float = 1.0,
) -> float:
    """Upper bound on L_alg(T) - L_batch(P*) — Eq. (32)."""
    if k_segments <= 0:
        return 0.0
    structural = max(penalty_c, tau * l_max)
    param = (dim_p / 2.0) * math.log(max(horizon_t / k_segments, 1.0))
    return k_segments * (structural + param + gamma)
