"""Hand-crafted traditional branch (VMAF proxy, Sec. III-C)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _ssim_map(x: Tensor, y: Tensor, window: int = 11) -> Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    c = x.shape[1]
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (num / den.clamp(min=1e-8)).mean(dim=1, keepdim=True)


def _detail_loss(x: Tensor, y: Tensor) -> Tensor:
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    c = x.shape[1]
    kx = kx.expand(c, 1, 3, 3)
    ky = ky.expand(c, 1, 3, 3)
    gx_x = F.conv2d(x, kx, padding=1, groups=c)
    gy_x = F.conv2d(x, ky, padding=1, groups=c)
    gx_y = F.conv2d(y, kx, padding=1, groups=c)
    gy_y = F.conv2d(y, ky, padding=1, groups=c)
    mag_x = torch.sqrt(gx_x * gx_x + gy_x * gy_x + 1e-8)
    mag_y = torch.sqrt(gx_y * gx_y + gy_y * gy_y + 1e-8)
    return (mag_x - mag_y).abs().mean(dim=1, keepdim=True)


def trad_vmaf_proxy(ref: Tensor, dist: Tensor) -> Tensor:
    """Fusion of SSIM, detail loss, and MSE cues (VMAF stand-in when libvmaf unavailable)."""
    if ref.dim() == 3:
        ref = ref.unsqueeze(0)
        dist = dist.unsqueeze(0)
    ssim = _ssim_map(ref, dist).mean()
    dlm = 1.0 - _detail_loss(ref, dist).mean().clamp(0.0, 1.0)
    mse = 1.0 - F.mse_loss(ref, dist).clamp(0.0, 1.0)
    score = 100.0 * (0.5 * ssim + 0.3 * dlm + 0.2 * mse)
    return score.reshape(1)
