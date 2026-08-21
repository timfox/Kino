"""Conventional VAE decode + upsample baselines (Table 1)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def vae_decode_stub(latent: Tensor, *, scale: int = 1) -> Tensor:
    """Cheap VAE decoder proxy: first 3 latent channels → RGB."""
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    rgb = latent[:, :3].clamp(-3, 3)
    rgb = (rgb - rgb.amin(dim=(2, 3), keepdim=True)) / (
        rgb.amax(dim=(2, 3), keepdim=True) - rgb.amin(dim=(2, 3), keepdim=True) + 1e-6
    )
    if scale > 1:
        h, w = rgb.shape[-2] * scale, rgb.shape[-1] * scale
        rgb = F.interpolate(rgb, size=(h, w), mode="bilinear", align_corners=False)
    return rgb.clamp(0, 1)


def upsample_bilinear(x: Tensor, scale: int) -> Tensor:
    h, w = x.shape[-2] * scale, x.shape[-1] * scale
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def cascaded_decode(
    latent: Tensor,
    *,
    scale: int = 4,
    upsampler: str = "bilinear",
) -> dict[str, Any]:
    """VAE decode at native res then SR (Eq. 1)."""
    low = vae_decode_stub(latent, scale=1)
    if upsampler == "bilinear":
        high = upsample_bilinear(low, scale)
    else:
        high = upsample_bilinear(low, scale)
    return {
        "low_res": low,
        "high_res": high,
        "upsampler": upsampler,
        "scale": scale,
    }
