"""Training losses (pixel + optional route alignment stub)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target)


def ssim_stub(pred: Tensor, target: Tensor) -> Tensor:
    """Structural term stub (1 − mean normalized correlation)."""
    p = pred.flatten(1)
    t = target.flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1)).clamp(min=1e-6)
    return 1.0 - (num / den).mean()
