"""Visual overlap gating (Sec. IV-C)."""

from __future__ import annotations

import math

from ltx_trainer.ground_texture_slam.schema import Pose2D


def fov_corners(pose: Pose2D, *, width: float = 0.6, height: float = 0.4) -> list[tuple[float, float]]:
    """Ground-plane FOV rectangle corners in world frame."""
    cx, cy, yaw = pose.x, pose.y, pose.yaw
    hw, hh = width / 2, height / 2
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    out = []
    c, s = math.cos(yaw), math.sin(yaw)
    for lx, ly in corners_local:
        wx = cx + c * lx - s * ly
        wy = cy + s * lx + c * ly
        out.append((wx, wy))
    return out


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _rect_intersection_area(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Axis-aligned bounding box overlap proxy for two quads."""
    ax = [p[0] for p in a]
    ay = [p[1] for p in a]
    bx = [p[0] for p in b]
    by = [p[1] for p in b]
    ix1 = max(min(ax), min(bx))
    iy1 = max(min(ay), min(by))
    ix2 = min(max(ax), max(bx))
    iy2 = min(max(ay), max(by))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def visual_overlap(pose_a: Pose2D, pose_b: Pose2D, *, min_area: float = 0.01) -> bool:
    """True if estimated FOV overlap exceeds threshold."""
    ca = fov_corners(pose_a)
    cb = fov_corners(pose_b)
    return _rect_intersection_area(ca, cb) >= min_area
