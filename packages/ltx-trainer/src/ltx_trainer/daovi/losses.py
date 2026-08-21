"""Inpainting reconstruction losses."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_reconstruction_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """L1 on masked regions only."""
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom
