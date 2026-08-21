"""Robust Multi-view Fusion — RMF (Sec. III-C, Eq. 6–7)."""

from __future__ import annotations

import math
from typing import Sequence


def _exp_g(x: float) -> float:
    return math.exp(x)


def rmf_weights(
    uncertainties: Sequence[float],
    conflicts: Sequence[float],
    *,
    training: bool = False,
) -> list[float]:
    r"""Fusion weights ``w_v``; training uses conflict-only (Eq. 7), test uses both (Eq. 6)."""
    scores: list[float] = []
    for u, o in zip(uncertainties, conflicts, strict=True):
        if training:
            scores.append(_exp_g(1.0 - o))
        else:
            scores.append(_exp_g((1.0 - u) * (1.0 - o)))
    total = sum(scores)
    if total < 1e-12:
        v = len(scores)
        return [1.0 / v] * v
    return [s / total for s in scores]


def fuse_memberships(
    memberships: Sequence[Sequence[float]],
    uncertainties: Sequence[float],
    conflicts: Sequence[float],
    *,
    training: bool = False,
) -> list[float]:
    r"""Fused membership ``m^a`` (Eq. 6–7)."""
    weights = rmf_weights(uncertainties, conflicts, training=training)
    k = len(memberships[0])
    fused = [0.0] * k
    for w, m in zip(weights, memberships, strict=True):
        for i in range(k):
            fused[i] += w * m[i]
    return fused
