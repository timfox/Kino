"""Standard FPS (Alg. 1) and RadiusFPS pruning helpers (§3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def l2_sq(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def l2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(l2_sq(a, b))


@dataclass
class Voxel:
    voxel_id: int
    center: tuple[float, float, float]
    point_indices: list[int]
    radius: float


def spherical_lower_bound(
    sj: tuple[float, float, float],
    ck: tuple[float, float, float],
    rk: float,
) -> float:
    """LB(vk, sj) = max(0, ||sj - ck||2 - rk) — Eq. 23."""
    return max(0.0, l2(sj, ck) - rk)


def should_prune_voxel(lb: float, dist_v: float) -> bool:
    """Safe voxel skip when LB >= Distv[k] — Eq. 25."""
    return lb >= dist_v


def should_skip_point(
    pi: tuple[float, float, float],
    sj: tuple[float, float, float],
    dist_p: float,
) -> bool:
    """Coordinate-wise point skip — Eq. 28."""
    return any(abs(pi[a] - sj[a]) >= dist_p for a in range(3))


def standard_fps(
    points: list[tuple[float, float, float]],
    m: int,
    *,
    seed_index: int = 0,
) -> list[int]:
    """Algorithm 1 — exact FPS with deterministic tie-breaking (smallest index)."""
    n = len(points)
    if m <= 0 or n == 0:
        return []
    m = min(m, n)
    dist = [float("inf")] * n
    selected: list[int] = []
    start = seed_index % n
    selected.append(start)
    dist[start] = 0.0
    for _ in range(1, m):
        last = selected[-1]
        slast = points[last]
        for i, pi in enumerate(points):
            d = l2_sq(pi, slast)
            if d < dist[i]:
                dist[i] = d
        best_i = max(range(n), key=lambda i: (dist[i], -i))
        selected.append(best_i)
    return selected


def update_distances_from_sample(
    points: list[tuple[float, float, float]],
    dist_p: list[float],
    sj_index: int,
) -> None:
    """Eq. 1 distance update from newly selected sample."""
    sj = points[sj_index]
    for i, pi in enumerate(points):
        d = l2_sq(pi, sj)
        if d < dist_p[i]:
            dist_p[i] = d


def argmax_dist(dist_p: list[float]) -> int:
    """Deterministic argmax: largest distance, smallest index on tie."""
    return max(range(len(dist_p)), key=lambda i: (dist_p[i], -i))


def fps_algorithm_card() -> dict[str, Any]:
    return {
        "complexity": "O(N · M)",
        "update_rule": "Distp[i] ← min(Distp[i], ||pi − slast||2)",
        "selection": "pnext ← argmax Distp with smallest-index tie-break",
        "ineffective_fraction": ">90% updates ineffective in later iterations (Fig. 3)",
    }
