"""Centroid arithmetic in x-vector space (Eq. 1, Sec. 3.1.4)."""

from __future__ import annotations

import math
from typing import Sequence


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i, val in enumerate(v):
            out[i] += val
    n = float(len(vectors))
    return [x / n for x in out]


def emotion_tau(
    emo_vectors: Sequence[Sequence[float]],
    neutral_vectors: Sequence[Sequence[float]],
) -> list[float]:
    """τ_emo = E[x(s, emo)] - E[x(s, neutral)]."""
    emo_c = centroid(emo_vectors)
    neu_c = centroid(neutral_vectors)
    return [e - n for e, n in zip(emo_c, neu_c, strict=True)]


def apply_tau(
    x_target_neutral: Sequence[float],
    tau: Sequence[float],
    alpha: float,
) -> list[float]:
    """x_new = x(target, neutral) + α · τ_emo."""
    return [x + alpha * t for x, t in zip(x_target_neutral, tau, strict=True)]


def interpolate(
    x_neutral: Sequence[float],
    x_emo: Sequence[float],
    alpha: float,
) -> list[float]:
    """Same-speaker interpolation: (1-α)x_neutral + α x_emo."""
    return [(1.0 - alpha) * n + alpha * e for n, e in zip(x_neutral, x_emo, strict=True)]


def l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = l2_norm(a)
    nb = l2_norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)
