"""FBImmersion gyration-circle IoU smoke (arXiv:2605.22521)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ltx_trainer.fbimmersion.config import FBImmersionConfig


@dataclass(frozen=True)
class _Circle:
    xc: float
    yc: float
    radius: float


def _intersection_area(c1: _Circle, c2: _Circle) -> float:
    r1, r2 = c1.radius, c2.radius
    if r1 <= 0.0 or r2 <= 0.0:
        return 0.0
    d = math.hypot(c1.xc - c2.xc, c1.yc - c2.yc)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        r_min = min(r1, r2)
        return math.pi * r_min * r_min
    part1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    part2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    part3 = 0.5 * math.sqrt(
        max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0.0)
    )
    return part1 + part2 - part3


def _immersion_pct(c1: _Circle, c2: _Circle) -> float:
    union = math.pi * c1.radius**2 + math.pi * c2.radius**2 - _intersection_area(c1, c2)
    if union <= 1e-12:
        return 100.0
    return _intersection_area(c1, c2) / union * 100.0


def evaluation_smoke(cfg: FBImmersionConfig | None = None) -> dict[str, Any]:
    c = cfg or FBImmersionConfig()
    cond = _Circle(0.0, 0.0, 1.0)
    gt = _Circle(0.15, 0.0, 1.0)
    imm = _immersion_pct(cond, gt)
    return {
        "paper": "arXiv:2605.22521",
        "immersion_pct": round(imm, 2),
    }
