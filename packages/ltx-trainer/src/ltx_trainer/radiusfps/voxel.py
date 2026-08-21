"""Spherical voxel preprocessing (§3.2–3.4)."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.radiusfps.constants import SPHERICAL_RADIUS_FACTOR
from ltx_trainer.radiusfps.fps import Voxel, l2, should_prune_voxel, should_skip_point, spherical_lower_bound


def voxel_side_length(extents: tuple[float, float, float], nvox: int, *, alpha: float = 1.001) -> float:
    """L = α max(Ex,Ey,Ez) / nvox — Eq. 7."""
    ex, ey, ez = extents
    return alpha * max(ex, ey, ez) / nvox


def voxel_index(
    p: tuple[float, float, float],
    pmin: tuple[float, float, float],
    L: float,
    nvox: int,
) -> tuple[int, int, int]:
    """Discrete voxel coords with clamp — Eq. 8."""
    out = []
    for a in range(3):
        idx = int(math.floor((p[a] - pmin[a]) / L))
        idx = max(0, min(nvox - 1, idx))
        out.append(idx)
    return out[0], out[1], out[2]


def flatten_vid(vx: int, vy: int, vz: int, nvox: int) -> int:
    """1D voxel id — Eq. 31."""
    return vz * nvox * nvox + vy * nvox + vx


def build_active_voxels(
    points: list[tuple[float, float, float]],
    *,
    nvox: int = 16,
    alpha: float = 1.001,
) -> tuple[list[Voxel], float, tuple[float, float, float], tuple[float, float, float]]:
    """Sparse active-voxel index (CPU stub)."""
    if not points:
        return [], 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    pmin = (min(xs), min(ys), min(zs))
    pmax = (max(xs), max(ys), max(zs))
    extents = (pmax[0] - pmin[0], pmax[1] - pmin[1], pmax[2] - pmin[2])
    L = voxel_side_length(extents, nvox, alpha=alpha)
    rk = SPHERICAL_RADIUS_FACTOR * L
    buckets: dict[int, list[int]] = {}
    for i, p in enumerate(points):
        vx, vy, vz = voxel_index(p, pmin, L, nvox)
        vid = flatten_vid(vx, vy, vz, nvox)
        buckets.setdefault(vid, []).append(i)
    voxels: list[Voxel] = []
    for vid, idxs in sorted(buckets.items()):
        pts = [points[j] for j in idxs]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        voxels.append(Voxel(voxel_id=vid, center=(cx, cy, cz), point_indices=idxs, radius=rk))
    return voxels, L, pmin, pmax


def sync_dist_v(voxels: list[Voxel], dist_p: list[float]) -> list[float]:
    """Distv[k] = max_{pi in vk} Distp[i] — Eq. 11/17."""
    return [max(dist_p[i] for i in v.point_indices) for v in voxels]


def radiusfps_update_iteration(
    points: list[tuple[float, float, float]],
    voxels: list[Voxel],
    dist_p: list[float],
    sj_index: int,
) -> tuple[list[float], int, int]:
    """One RadiusFPS iteration: prune voxels, point-skip, exact update."""
    sj = points[sj_index]
    dist_v = sync_dist_v(voxels, dist_p)
    pruned = 0
    skipped_points = 0
    updated_points = 0
    for k, vk in enumerate(voxels):
        lb = spherical_lower_bound(sj, vk.center, vk.radius)
        if should_prune_voxel(lb, dist_v[k]):
            pruned += 1
            continue
        for i in vk.point_indices:
            pi = points[i]
            if should_skip_point(pi, sj, dist_p[i]):
                skipped_points += 1
                continue
            d = l2(pi, sj)
            if d < dist_p[i]:
                dist_p[i] = d
                updated_points += 1
    return dist_p, pruned, skipped_points + updated_points


def voxel_method_card() -> dict[str, Any]:
    return {
        "voxel_side": "L = α max(E) / nvox",
        "sphere_radius": "rk = (√3/2) L",
        "lower_bound": "LB = max(0, ||sj−ck||2 − rk)",
        "prune_when": "LB(vk,sj) ≥ Distv[k]",
        "point_skip": "∃ axis a: |pa_i − sa_j| ≥ Distp[i]",
        "conservative": "preserves exact FPS under fixed seed + tie-break",
    }
