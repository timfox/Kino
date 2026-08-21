"""PSNR / SSIM / LPIPS stubs for NVS evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_proxy(pred: Tensor, target: Tensor) -> Tensor:
    return 1.0 - F.l1_loss(pred, target)


def lpips_proxy(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target).clamp(0, 1)
