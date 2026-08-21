"""Berhu loss for depth (PanoFormer protocol, Sec. 5.2)."""

from __future__ import annotations

import torch
from torch import Tensor

BERHU_T = 0.2


def berhu_loss(pred: Tensor, target: Tensor, *, t: float = BERHU_T) -> Tensor:
    diff = torch.abs(pred - target)
    c = t * diff.max().clamp(min=1e-6)
    return torch.where(diff <= c, diff, (diff.pow(2) + c**2) / (2 * c)).mean()
