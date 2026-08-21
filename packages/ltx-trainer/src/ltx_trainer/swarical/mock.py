"""Swarical alignment smoke."""

from __future__ import annotations

import random
from typing import Any

from ltx_trainer.swarical.localization import center_align
from ltx_trainer.swarical.planner import Point3


def grid_points(rows: int, cols: int, *, spacing_cm: float = 7.0) -> list[Point3]:
    pts: list[Point3] = []
    for r in range(rows):
        for c in range(cols):
            pts.append(Point3(c * spacing_cm, r * spacing_cm, 0.0))
    return pts


def perturbed_grid(*, noise_cm: float = 0.2) -> tuple[list[Point3], list[Point3]]:
    base = grid_points(4, 4)
    est = [
        Point3(
            p.x + random.uniform(-noise_cm, noise_cm),
            p.y + random.uniform(-noise_cm, noise_cm),
            p.z,
        )
        for p in base
    ]
    return base, est


def evaluation_smoke() -> dict[str, Any]:
    base, est = perturbed_grid(noise_cm=0.1)
    aligned = center_align(est, base)
    dx = sum(abs(a.x - b.x) for a, b in zip(aligned, base, strict=True)) / len(base)
    return {"num_points": len(base), "mean_dx_after_align": round(dx, 4)}
