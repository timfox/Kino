"""NVS metrics (PSNR, SSIM stub)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(pred: Tensor, gt: Tensor, eps: float = 1e-8) -> Tensor:
    mse = F.mse_loss(pred, gt)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


def ssim_stub(pred: Tensor, gt: Tensor) -> Tensor:
    p = pred.flatten()
    g = gt.flatten()
    c = ((p - p.mean()) * (g - g.mean())).mean()
    return (2 * c + 1e-3) / (p.var() + g.var() + 1e-3)
