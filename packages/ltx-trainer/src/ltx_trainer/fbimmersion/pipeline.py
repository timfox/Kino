"""FBI evaluation pipeline glue (Sec. 2–4)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.fbimmersion.config import (
    ACTIVITIES,
    CONDITIONS,
    ActivityName,
    ConditionName,
    FBImmersionConfig,
)
from ltx_trainer.fbimmersion.gyration import GyrationCircle, gyration_circle_from_submetrics
from ltx_trainer.fbimmersion.immersion import immersion_index_percent
from ltx_trainer.fbimmersion.miros import platform_spec, scale_ladder
from ltx_trainer.fbimmersion.submetrics import (
    SUBMETRICS_BY_ACTIVITY,
    extract_submetric_values,
)


def compute_immersion_for_activity(
    activity: ActivityName,
    condition_series: dict[ConditionName, dict[str, Tensor]],
    *,
    ground_truth_key: ConditionName = "ground_truth",
) -> dict[str, Any]:
    """Extract submetrics → gyration circles → immersion index per condition."""
    gt_iqr = extract_submetric_values(activity, condition_series[ground_truth_key])
    gt_circle = gyration_circle_from_submetrics(gt_iqr)
    results: dict[str, Any] = {
        "activity": activity,
        "submetrics": list(SUBMETRICS_BY_ACTIVITY[activity]),
        "ground_truth": {"submetric_iqr": gt_iqr, "circle": _circle_dict(gt_circle)},
        "conditions": {},
    }
    for cond, series in condition_series.items():
        if cond == ground_truth_key:
            continue
        iqr = extract_submetric_values(activity, series)
        circle = gyration_circle_from_submetrics(iqr)
        idx = immersion_index_percent(circle, gt_circle)
        results["conditions"][cond] = {
            "submetric_iqr": iqr,
            "circle": _circle_dict(circle),
            "immersion_index_pct": idx,
        }
    return results


def _circle_dict(c: GyrationCircle) -> dict[str, float]:
    return {"xc": c.xc, "yc": c.yc, "radius": c.radius}


def demo_skiing_series(
    *,
    frames: int = 600,
    seed: int = 0,
) -> dict[ConditionName, dict[str, Tensor]]:
    """Synthetic skiing submetrics: platform closer to ground truth than no-feedback."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * 3.14159, frames)

    def _make(scale: float, knee_amp: float) -> dict[str, Tensor]:
        cog = scale * 0.02 * torch.sin(t) + 0.005 * torch.randn(frames, generator=g)
        rk = knee_amp * 15 * torch.sin(t * 1.1) + 2 * torch.randn(frames, generator=g)
        lk = knee_amp * 14 * torch.sin(t * 0.9) + 2 * torch.randn(frames, generator=g)
        return {"cog_change": cog, "right_knee_angle": rk, "left_knee_angle": lk}

    return {
        "ground_truth": _make(1.0, 1.0),
        "platform": _make(0.85, 0.75),
        "no_feedback": _make(0.3, 0.25),
    }


def demo_boating_standing_series(
    *,
    frames: int = 400,
    seed: int = 1,
) -> dict[ConditionName, dict[str, Tensor]]:
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 2 * 3.14159, frames)

    def _make(sway: float, trunk: float) -> dict[str, Tensor]:
        xy = torch.stack(
            [
                sway * 0.05 * torch.sin(t) + 0.01 * torch.randn(frames, generator=g),
                sway * 0.04 * torch.cos(t) + 0.01 * torch.randn(frames, generator=g),
            ],
            dim=-1,
        )
        chest = trunk * 12 * torch.sin(t * 0.7) + torch.randn(frames, generator=g)
        feet = trunk * 8 * torch.sin(t * 0.5 + 0.3) + torch.randn(frames, generator=g)
        return {"cog_sway_xy": xy, "cog_chest_angle": chest, "cog_feet_angle": feet}

    return {
        "ground_truth": _make(1.0, 1.0),
        "platform": _make(0.7, 0.65),
        "no_feedback": _make(0.05, 0.05),
    }


def reference_immersion_table(cfg: FBImmersionConfig | None = None) -> dict[str, dict[str, float]]:
    """Paper-reported immersion indices (Sec. 2, Fig. 3)."""
    cfg = cfg or FBImmersionConfig()
    return {
        "skiing": {
            "no_feedback": cfg.skiing_no_feedback_pct,
            "platform": cfg.skiing_platform_pct,
        },
        "boating": {
            "no_feedback": cfg.boating_no_feedback_pct,
            "platform": cfg.boating_platform_pct,
        },
    }


def acceleration_tracking_table(cfg: FBImmersionConfig | None = None) -> dict[str, float]:
    """Spearman ρ from Fig. 5."""
    cfg = cfg or FBImmersionConfig()
    return {
        "video_platform_rho_x": cfg.video_platform_rho_x,
        "video_platform_rho_y": cfg.video_platform_rho_y,
        "sim_platform_rho": cfg.sim_platform_rho,
    }


def immersion_layers() -> list[dict[str, str]]:
    """Three-level immersion framework (Fig. 1)."""
    return [
        {"level": "audio_visual", "description": "Spatial audio + high-resolution VR rendering"},
        {"level": "physical", "description": "Localized haptic gloves and wearables"},
        {"level": "full_body", "description": "Whole-body kinetic interaction via MIROS surfaces"},
    ]


def dataset_card(cfg: FBImmersionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FBImmersionConfig()
    return {
        "name": "FBImmersion",
        "paper": "arXiv:2605.22521",
        "title": "Quantifying Full-Body Immersion",
        "task": "immersion index from task-specific body submetrics",
        "platform": "MIROS",
        "activities": list(ACTIVITIES),
        "conditions": list(CONDITIONS),
        "min_submetrics": cfg.min_submetrics,
        "platform_spec": platform_spec(cfg).__dict__,
        "scale_ladder": scale_ladder(),
        "immersion_layers": immersion_layers(),
    }
