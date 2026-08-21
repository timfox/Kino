"""MIROS platform and control modes (Sec. 2, Fig. 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.fbimmersion.config import FBImmersionConfig, OperatingMode


@dataclass(frozen=True)
class MIROSPlatformSpec:
    """Full-body-scale MIROS module specifications."""

    dof: int
    bandwidth_hz: float
    peak_velocity_mm_s: float
    torque_nm: float
    human_scale_cm: float


def platform_spec(cfg: FBImmersionConfig | None = None) -> MIROSPlatformSpec:
    cfg = cfg or FBImmersionConfig()
    return MIROSPlatformSpec(
        dof=cfg.platform_dof,
        bandwidth_hz=cfg.actuation_bandwidth_hz,
        peak_velocity_mm_s=cfg.peak_velocity_mm_s,
        torque_nm=cfg.actuator_torque_nm,
        human_scale_cm=cfg.human_scale_cm,
    )


def scale_ladder() -> list[dict[str, Any]]:
    """Five interaction scales from needle to full-body (Fig. 6)."""
    return [
        {"scale": "needle", "size": "1–8 mm", "manufacturing": "origami", "materials": ["kapton", "fr4"]},
        {"scale": "fingertip", "size": "~5 cm", "manufacturing": "origami", "materials": ["kapton", "glass_fiber"]},
        {"scale": "hand", "size": "~5 cm", "manufacturing": "origami", "materials": ["kapton", "fr4"]},
        {"scale": "arm", "size": "dm", "manufacturing": "hybrid", "materials": ["wood", "polymer"]},
        {"scale": "full_body", "size": "35 cm–3 m", "manufacturing": "conventional", "materials": ["aluminum", "steel"]},
    ]


def operating_mode_description(mode: OperatingMode) -> str:
    if mode == "feedback_rendering":
        return (
            "Cameras/IMUs record real motion; controller renders position, velocity, "
            "and acceleration on MIROS (e.g. skiing: vertical accel + pitch/roll mapping)."
        )
    return (
        "User applies force to platform to control virtual avatar; controller computes "
        "environment dynamics and renders feedback through MIROS."
    )
