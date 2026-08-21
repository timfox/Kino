"""Spatial-temporal anchor degradation for Stage-II training (Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def degrade_anchor(
    frame: Tensor,
    *,
    blur_kernel: int = 5,
    downscale: float = 0.25,
    noise_std: float = 0.02,
) -> Tensor:
    """Strip HF details: blur → down/up → Gaussian noise."""
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
    x = frame
    if blur_kernel > 1:
        k = blur_kernel
        pad = k // 2
        x = F.avg_pool2d(F.pad(x, [pad] * 4, mode="reflect"), k, stride=1)
    h, w = x.shape[-2:]
    small = F.interpolate(x, scale_factor=downscale, mode="bilinear", align_corners=False)
    x = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x.clamp(0.0, 1.0)
