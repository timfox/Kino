"""Configuration for Full-Body Immersion quantification (Bakir et al., arXiv:2605.22521)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActivityName = Literal["skiing", "boating_standing", "boating_seated"]
ConditionName = Literal["ground_truth", "platform", "no_feedback"]
OperatingMode = Literal["feedback_rendering", "interactive"]

ACTIVITIES: tuple[ActivityName, ...] = ("skiing", "boating_standing", "boating_seated")
CONDITIONS: tuple[ConditionName, ...] = ("ground_truth", "platform", "no_feedback")


@dataclass
class FBImmersionConfig:
    """Defaults from paper Sec. 2–4."""

    min_submetrics: int = 3
    # MIROS full-body platform (Sec. 2)
    platform_dof: int = 3
    actuation_bandwidth_hz: float = 20.0
    peak_velocity_mm_s: float = 120.0
    actuator_torque_nm: float = 892.0
    human_scale_cm: float = 30.0
    # Paper Table / Fig. 3 immersion indices (%)
    skiing_no_feedback_pct: float = 14.0
    skiing_platform_pct: float = 23.0
    boating_no_feedback_pct: float = 1.0
    boating_platform_pct: float = 20.0
    # Acceleration tracking Spearman ρ (Fig. 5)
    video_platform_rho_x: float = 0.8
    video_platform_rho_y: float = 0.65
    sim_platform_rho: float = 0.6
