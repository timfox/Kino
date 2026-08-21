"""Pose AUC proxy metrics (Sec. 5.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def angular_error_deg(pred: Tensor, tgt: Tensor) -> Tensor:
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt = tgt / tgt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (pred * tgt).sum(dim=-1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def auc_at_threshold(errors_deg: Tensor, thresh: float) -> float:
    return float((errors_deg < thresh).float().mean().item())
