"""PSNR / SSIM for LIE benchmarks (Tables 2–3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(pred: Tensor, gt: Tensor, *, max_val: float = 1.0) -> float:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    mse = F.mse_loss(pred.detach(), gt.detach())
    if mse.item() <= 0:
        return 99.0
    ratio = (max_val**2) / mse.detach().clamp_min(1e-12)
    return float((10.0 * torch.log10(ratio)).item())


def ssim(pred: Tensor, gt: Tensor) -> float:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    c = pred.shape[1]
    window = 11
    pad = window // 2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(gt, window, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, window, stride=1, padding=pad) - mu_x**2
    sigma_y = F.avg_pool2d(gt * gt, window, stride=1, padding=pad) - mu_y**2
    sigma_xy = F.avg_pool2d(pred * gt, window, stride=1, padding=pad) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(ssim_map.mean().item())
