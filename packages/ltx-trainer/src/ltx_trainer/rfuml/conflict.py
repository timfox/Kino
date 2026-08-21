"""Inter-view conflict (Sec. III-C, Eq. 5)."""

from __future__ import annotations

import math
from typing import Sequence


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def conflict_for_view(memberships: Sequence[Sequence[float]], v: int) -> float:
    r"""Conflict of view ``v`` relative to others (Eq. 5)."""
    v_count = len(memberships)
    if v_count <= 1:
        return 0.0
    mv = memberships[v]
    nv = _norm(mv)
    if nv < 1e-12:
        return 0.0
    acc = 0.0
    for j in range(v_count):
        if j == v:
            continue
        mj = memberships[j]
        nj = _norm(mj)
        if nj < 1e-12:
            sim = 0.0
        else:
            sim = _dot(mv, mj) / (nv * nj)
        acc += 1.0 - sim
    return acc / (v_count - 1)
