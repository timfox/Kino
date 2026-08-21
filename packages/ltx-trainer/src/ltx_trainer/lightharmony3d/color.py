"""sRGB / linear color utilities for compositing (Sec. 3.6)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.lightharmony3d.config import FUSION_GAMMA, RGB_LUMA_WEIGHTS


def srgb_to_linear(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: Tensor) -> Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def linear_luminance(rgb: Tensor) -> Tensor:
    w = rgb.new_tensor(RGB_LUMA_WEIGHTS).view(1, 3, 1, 1)
    if rgb.shape[-3] != 3:
        w = w.squeeze(0)
    lin = srgb_to_linear(rgb) if rgb.max() <= 1.0 + 1e-6 else rgb
    return (lin * w).sum(dim=-3, keepdim=True)


def normalized_luminance_ldr(img: Tensor, gamma: float = FUSION_GAMMA) -> Tensor:
    """Eq. (1): L̃ = (w^T I)^γ on sRGB inputs."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    weighted = (img * img.new_tensor(RGB_LUMA_WEIGHTS).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return weighted.clamp(min=1e-8).pow(gamma)
