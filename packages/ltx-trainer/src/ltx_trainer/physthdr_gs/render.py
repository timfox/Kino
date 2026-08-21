"""Differentiable Gaussian splatting stub for HDR images (Sec. 3.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def splat_gaussians(
    xyz: Tensor,
    colors: Tensor,
    opacity: Tensor,
    *,
    image_size: int = 64,
) -> Tensor:
    """
    Project 3D Gaussians to an orthographic HDR image via soft splatting.

    Args:
        xyz: [N, 3] positions in [-1, 1]
        colors: [N, 3] HDR radiance
        opacity: [N, 1] in (0, 1)
    Returns:
        HDR image [1, 3, H, W]
    """
    n = xyz.shape[0]
    device = xyz.device
    h = w = image_size
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2)

    xy = xyz[:, :2].unsqueeze(0)  # [1, N, 2]
    sigma = 0.08 + 0.02 * torch.rand(n, 1, device=device)
    dist2 = ((grid.unsqueeze(2) - xy.unsqueeze(1)) ** 2).sum(dim=-1)
    weights = torch.exp(-dist2 / (2 * sigma.T**2)) * opacity.T  # [1, HW, N]
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    cols = colors.unsqueeze(0).unsqueeze(1)  # [1, 1, N, 3]
    img = (weights.unsqueeze(-1) * cols).sum(dim=2)
    return img.reshape(1, 3, h, w)


def gaussian_blur(img: Tensor, kernel_size: int = 5) -> Tensor:
    """Gaussian blur for HDR consistency loss (Eq. 16)."""
    if kernel_size <= 1:
        return img
    pad = kernel_size // 2
    x = torch.arange(kernel_size, device=img.device, dtype=img.dtype) - pad
    g = torch.exp(-0.5 * (x / max(pad, 1)) ** 2)
    g = g / g.sum()
    kh = g.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    kv = g.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    img = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    img = F.conv2d(img, kh, groups=3)
    img = F.conv2d(img, kv, groups=3)
    return img
