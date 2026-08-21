"""Dead reckoning dispatch error model (companion Sec. 2.1, original paper Sec. 5)."""

from __future__ import annotations

import math
import random

from ltx_trainer.swarical.planner import Point3


def apply_dead_reckoning(
    ground_truth: list[Point3],
    *,
    alpha_deg: float,
    rng: random.Random | None = None,
) -> list[Point3]:
    """Dispatch each FLS from origin toward its target with random angular error ≤ ``alpha_deg``.

    Companion notebook visualizes ground truth vs dead-reckoned point clouds before localization.
    """
    rng = rng or random.Random()
    alpha_rad = math.radians(alpha_deg)
    out: list[Point3] = []
    for target in ground_truth:
        dist = math.sqrt(target.x**2 + target.y**2 + target.z**2)
        if dist < 1e-9:
            out.append(Point3(0.0, 0.0, 0.0))
            continue
        base_angle = math.atan2(target.y, target.x)
        err = rng.uniform(-alpha_rad, alpha_rad)
        angle = base_angle + err
        # Preserve radial distance; perturb heading only (2D dominant in small-scale grid).
        out.append(
            Point3(
                dist * math.cos(angle),
                dist * math.sin(angle),
                target.z,
            )
        )
    return out
