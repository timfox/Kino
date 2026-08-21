"""RMSE and paper Table I metrics (Sec. V-B)."""

from __future__ import annotations

import math

from ltx_trainer.ground_texture_slam.schema import LoopClosureMethod, Pose2D


def position_rmse(est: list[Pose2D], gt: list[Pose2D]) -> float:
    if not est or not gt:
        return float("inf")
    n = min(len(est), len(gt))
    err = 0.0
    for i in range(n):
        dx = est[i].x - gt[i].x
        dy = est[i].y - gt[i].y
        err += dx * dx + dy * dy
    return math.sqrt(err / n)


def orientation_rmse_deg(est: list[Pose2D], gt: list[Pose2D]) -> float:
    if not est or not gt:
        return float("inf")
    n = min(len(est), len(gt))
    err = 0.0
    for i in range(n):
        d = est[i].yaw - gt[i].yaw
        err += d * d
    return math.degrees(math.sqrt(err / n))


def table_i_rmse() -> dict[str, dict[str, float | str]]:
    """Table I reported RMSE (Sec. V-B)."""
    return {
        LoopClosureMethod.KLD.value: {"position_m": 0.086, "orientation_deg": 1.572},
        LoopClosureMethod.KLD_GRAY.value: {"position_m": 0.085, "orientation_deg": 1.530},
        LoopClosureMethod.VISUAL_OVERLAP.value: {"position_m": 0.212, "orientation_deg": 8.153},
        LoopClosureMethod.JIH.value: {"position_m": 0.324, "orientation_deg": 19.629},
        "orb_slam3": {"position_m": "N/A", "orientation_deg": "N/A"},
        LoopClosureMethod.ORIGINAL_SINGLE.value: {"position_m": 0.540, "orientation_deg": 27.013},
        LoopClosureMethod.ORIGINAL_MANY.value: {"position_m": 0.266, "orientation_deg": 14.963},
        LoopClosureMethod.ODOMETRY.value: {"position_m": 0.210, "orientation_deg": 7.244},
    }
