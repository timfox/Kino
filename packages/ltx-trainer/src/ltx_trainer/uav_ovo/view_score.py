"""Geometry-derived view scores and split assignment (Sec. 3.1)."""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

from ltx_trainer.uav_ovo.config import UAVOVOConfig


class ViewSplit(str, Enum):
    TRAIN = "train"
    ID_TEST = "id_test"
    ISOLATION = "isolation"
    OOD_TEST = "ood_test"


def pitch_offset_deg(theta_deg: Tensor) -> Tensor:
    """View score s = θ − 90° (degrees)."""
    return theta_deg - 90.0


def angle_optical_axis_to_normal(optical_axis: Tensor, ground_normal: Tensor, eps: float = 1e-6) -> Tensor:
    """θ = arccos(clip(o^T n, -1, 1)) in degrees; inputs (..., 3)."""
    o = optical_axis / (optical_axis.norm(dim=-1, keepdim=True) + eps)
    n = ground_normal / (ground_normal.norm(dim=-1, keepdim=True) + eps)
    cos = (o * n).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def video_view_score(frame_scores: Tensor) -> Tensor:
    """Median over valid frame scores."""
    return frame_scores.median()


def assign_split(view_score: float, cfg: UAVOVOConfig | None = None) -> ViewSplit | None:
    """Map scalar view score to benchmark split."""
    cfg = cfg or UAVOVOConfig()
    if view_score <= cfg.train_view_max:
        return ViewSplit.TRAIN  # also used for ID pool; caller distinguishes by protocol
    if cfg.isolation_view_min < view_score <= cfg.isolation_view_max:
        return ViewSplit.ISOLATION
    if view_score > cfg.ood_view_min:
        return ViewSplit.OOD_TEST
    return None


def group_median_score(video_scores: list[float]) -> float:
    """Timestamp-group median (manual review assumed upstream)."""
    if not video_scores:
        return float("nan")
    t = torch.tensor(video_scores, dtype=torch.float64)
    return float(t.median())
