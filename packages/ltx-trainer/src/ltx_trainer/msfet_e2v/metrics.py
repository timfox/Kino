"""Full-reference metrics used in Table I (PSNR, SSIM, LPIPS ordering)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(pred: Tensor, target: Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < eps:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def ssim_proxy(pred: Tensor, target: Tensor, window: int = 11) -> float:
    """Simplified SSIM for smoke tests (not full Wang SSIM)."""
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=window // 2)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=window // 2)
    sigma_x = F.avg_pool2d(pred * pred, window, stride=1, padding=window // 2) - mu_x**2
    sigma_y = F.avg_pool2d(target * target, window, stride=1, padding=window // 2) - mu_y**2
    sigma_xy = F.avg_pool2d(pred * target, window, stride=1, padding=window // 2) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float((num / den).mean().clamp(0, 1).item())
