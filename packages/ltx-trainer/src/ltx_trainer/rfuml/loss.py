"""Category credibility learning losses (Sec. III-D, Eq. 9, 11, 16)."""

from __future__ import annotations

import math
from typing import Sequence

from ltx_trainer.rfuml.fuzzy import training_credibility


def _bce_pair(r: float, y: float) -> float:
    r = min(max(r, 1e-12), 1.0 - 1e-12)
    y = min(max(y, 0.0), 1.0)
    return -(y * math.log(r) + (1.0 - y) * math.log(1.0 - r))


def lccl(
    memberships: Sequence[float],
    labels: Sequence[float],
    *,
    use_training_credibility: bool = True,
) -> float:
    r"""Category credibility learning loss ``L_ccl`` (Eq. 9)."""
    if use_training_credibility:
        r_vec = training_credibility(memberships, labels)
    else:
        from ltx_trainer.rfuml.fuzzy import category_credibility

        r_vec = category_credibility(memberships)
    return sum(_bce_pair(rv, yv) for rv, yv in zip(r_vec, labels, strict=True)) / len(labels)


def lrccl(
    memberships: Sequence[float],
    labels: Sequence[float],
    importance: float,
) -> float:
    r"""Weighted ``L_rccl`` for conflicting views (Eq. 16)."""
    return importance * lccl(memberships, labels, use_training_credibility=True)


def total_loss(
    view_memberships: Sequence[Sequence[float]],
    fused_memberships: Sequence[float],
    labels: Sequence[float],
    *,
    gamma: float = 1.0,
    view_importance: Sequence[float] | None = None,
) -> float:
    r"""``L_total = γ L_ccl(r^a, y) + Σ_v L_ccl`` or robust variant (Eq. 11, 16)."""
    imp = view_importance or [1.0] * len(view_memberships)
    fused_term = gamma * lccl(fused_memberships, labels)
    view_term = 0.0
    for m, d in zip(view_memberships, imp, strict=True):
        if d >= 1.0 - 1e-9:
            view_term += lccl(m, labels)
        else:
            view_term += lrccl(m, labels, d)
    return fused_term + view_term
