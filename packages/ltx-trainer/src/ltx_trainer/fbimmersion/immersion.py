"""Immersion index via gyration-circle IoU (Eq. 1, Algorithm 1 lines 16–22)."""

from __future__ import annotations

import math

from ltx_trainer.fbimmersion.gyration import GyrationCircle


def circle_intersection_area(c1: GyrationCircle, c2: GyrationCircle) -> float:
    """Intersection area of two circles (Venn diagram, Sec. 4)."""
    r1, r2 = c1.radius, c2.radius
    if r1 <= 0.0 or r2 <= 0.0:
        return 0.0
    d = math.hypot(c1.xc - c2.xc, c1.yc - c2.yc)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        r_min = min(r1, r2)
        return math.pi * r_min * r_min
    # lens area
    part1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    part2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    part3 = 0.5 * math.sqrt(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    )
    return part1 + part2 - part3


def circle_union_area(c1: GyrationCircle, c2: GyrationCircle) -> float:
    """A_union = πr1² + πr2² − A_inter (Algorithm 1, line 20)."""
    a1 = math.pi * c1.radius * c1.radius
    a2 = math.pi * c2.radius * c2.radius
    inter = circle_intersection_area(c1, c2)
    return a1 + a2 - inter


def immersion_index_percent(
    condition: GyrationCircle,
    ground_truth: GyrationCircle,
) -> float:
    """Immersion (%) = A_inter / A_union × 100 (Eq. 1)."""
    union = circle_union_area(condition, ground_truth)
    if union <= 1e-12:
        return 100.0 if condition.radius <= 1e-12 and ground_truth.radius <= 1e-12 else 0.0
    inter = circle_intersection_area(condition, ground_truth)
    return float(inter / union * 100.0)
