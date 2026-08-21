"""Hausdorff and Chamfer distance for localization quality (Sec. 5.3)."""

from __future__ import annotations

import math

from ltx_trainer.swarical.planner import Point3
from ltx_trainer.swarical.localization import center_align


def _min_dist(a: Point3, cloud: list[Point3]) -> float:
    return min(a.distance_to(b) for b in cloud)


def hausdorff_distance(estimated: list[Point3], planner: list[Point3]) -> float:
    """Symmetric Hausdorff after centroid alignment (lower is better)."""
    if not estimated or not planner:
        return float("inf")
    e = center_align(estimated, planner)
    forward = max(_min_dist(p, planner) for p in e)
    backward = max(_min_dist(p, e) for p in planner)
    return max(forward, backward)


def chamfer_distance(estimated: list[Point3], planner: list[Point3]) -> float:
    r"""Chamfer(A,B) = (1/|A|)Σ min‖a−b‖² + (1/|B|)Σ min‖b−a‖² (Sec. 5.3 footnote)."""
    if not estimated or not planner:
        return float("inf")
    e = center_align(estimated, planner)
    term_a = sum(_min_dist(p, planner) ** 2 for p in e) / len(e)
    term_b = sum(_min_dist(p, e) ** 2 for p in planner) / len(planner)
    return term_a + term_b


def shrink_point_cloud(points: list[Point3], pct: float) -> list[Point3]:
    """Scale about centroid by ``(1 − ε/100)`` for error propagation model (Sec. 5.5)."""
    if not points:
        return []
    cx = sum(p.x for p in points) / len(points)
    cy = sum(p.y for p in points) / len(points)
    cz = sum(p.z for p in points) / len(points)
    scale = 1.0 - pct / 100.0
    return [
        Point3(cx + (p.x - cx) * scale, cy + (p.y - cy) * scale, cz + (p.z - cz) * scale)
        for p in points
    ]


def estimate_hd_from_camera_error(
    planner: list[Point3],
    *,
    pct_error: float,
) -> float:
    """Analytical HD proxy: shrink planner cloud by camera percentage error (Sec. 5.5)."""
    shrunk = shrink_point_cloud(planner, pct_error)
    return hausdorff_distance(shrunk, planner)
