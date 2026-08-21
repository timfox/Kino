"""Occupancy and video evaluation helpers (Sec. 4, supplement B.4)."""

from __future__ import annotations

import torch
from torch import Tensor


def binary_iou(pred_occ: Tensor, gt_occ: Tensor) -> float:
    """Geometric IoU: occupied vs free (any non-zero class)."""
    p = pred_occ.reshape(-1) > 0
    g = gt_occ.reshape(-1) > 0
    inter = (p & g).sum().float()
    union = (p | g).sum().float().clamp(min=1.0)
    return float((inter / union).item())


def mean_iou(pred: Tensor, gt: Tensor, num_classes: int, *, ignore_free: bool = True) -> float:
    """Per-class IoU averaged over non-free classes."""
    ious: list[float] = []
    for c in range(num_classes):
        if ignore_free and c == 0:
            continue
        p = pred == c
        g = gt == c
        if g.sum() == 0:
            continue
        inter = (p & g).sum().float()
        union = (p | g).sum().float().clamp(min=1.0)
        ious.append(float((inter / union).item()))
    return sum(ious) / max(len(ious), 1)


def bev_topdown(labels: Tensor) -> Tensor:
    """Lowest non-free class per column (supplement B.4)."""
    h, w, z = labels.shape[-3], labels.shape[-2], labels.shape[-1]
    flat = labels.reshape(*labels.shape[:-3], h, w, z)
    occ = flat > 0
    has = occ.any(dim=-1)
    idx = occ.float().argmax(dim=-1)
    return torch.where(has, idx, torch.zeros_like(idx))
