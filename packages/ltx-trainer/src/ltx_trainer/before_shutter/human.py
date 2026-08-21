"""Human subject + actionability metrics (Sec. 3.1, Eq. 6–7)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ltx_trainer.before_shutter.config import BeforeShutterConfig, PortraitPlanState


@dataclass
class SupportSample:
    x: float
    y: float
    z: float
    bone: str


def smpl_support_polygon(state: PortraitPlanState) -> list[tuple[float, float]]:
    """Projected support hull Q_s stub for balance metric."""
    x, y, z = state.root_translation
    w = 0.25
    return [(x - w, y - w), (x + w, y - w), (x + w, y + w), (x - w, y + w)]


def center_of_mass_xy(state: PortraitPlanState) -> tuple[float, float]:
    """Anthropometric CoM proxy (Winter 2009 stub)."""
    x, y, _ = state.root_translation
    return (x, y + 0.02)


def point_in_convex_hull(p: tuple[float, float], hull: list[tuple[float, float]]) -> bool:
    if len(hull) < 3:
        return True
    sign = None
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        cross = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
        if abs(cross) < 1e-9:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def balance_score(state: PortraitPlanState, *, cfg: BeforeShutterConfig | None = None) -> float:
    """S(s) in Eq. (7) — 1 if CoM projects inside support hull."""
    _ = cfg
    hull = smpl_support_polygon(state)
    com = center_of_mass_xy(state)
    return 1.0 if point_in_convex_hull(com, hull) else 0.0


def collision_penetration(state: PortraitPlanState, occupancy_z_floor: float = 0.0) -> int:
    """P_skel = 0 if no penetration — bone samples vs occupancy stub."""
    z = state.root_translation[2]
    return 1 if z < occupancy_z_floor - 0.05 else 0


def realize_staging(
    state: PortraitPlanState,
    anchor: str,
    *,
    cfg: BeforeShutterConfig | None = None,
) -> PortraitPlanState:
    """Actor: affordance-grounded placement stub."""
    _ = cfg
    if anchor in ("red_chair", "chair", "seat"):
        state.root_translation = (0.0, 0.0, 0.45)
        state.body_pose = (0.1, -0.2, 0.0, 0.3, 0.0, 0.0)
    elif anchor in ("floor", "stand"):
        state.root_translation = (0.0, 0.0, 0.0)
    else:
        state.root_translation = (0.2, 0.0, 0.0)
    return state


def batch_actionability(
    states: list[PortraitPlanState],
    *,
    cfg: BeforeShutterConfig | None = None,
) -> dict[str, float]:
    """R_coll, R_bal from Eq. (6–7)."""
    _ = cfg
    n = max(len(states), 1)
    r_coll = sum(1.0 if collision_penetration(s) == 0 else 0.0 for s in states) / n
    r_bal = sum(balance_score(s) for s in states) / n
    return {"R_coll": r_coll, "R_bal": r_bal}
