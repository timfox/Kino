"""Lightweight VAE codec stub for LatentHDR l2h without FLUX (posterior mean path)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StubVaeLatentCodec(nn.Module):
    """Conv encoder/decoder approximating μ(x) for tests and CPU inference."""

    def __init__(self, *, latent_channels: int = 16, scale: int = 4) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.scale = scale
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, stride=2, padding=1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def encode(self, rgb: Tensor) -> Tensor:
        """``rgb`` ``[B,3,H,W]`` or ``[3,H,W]`` → latent ``[B,C,h,w]`` or ``[C,h,w]``."""
        squeeze = False
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            squeeze = True
        z = self.enc(rgb.clamp(0.0, 1.0))
        return z.squeeze(0) if squeeze else z

    def decode(self, z: Tensor, *, out_hw: tuple[int, int] | None = None) -> Tensor:
        """Latent → RGB in ``[0,1]`` at ``out_hw`` or ``scale``× upsampled size."""
        squeeze = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeeze = True
        x = self.dec(z)
        if out_hw is not None:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        out = x.clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out

    def encode_video(self, rgb_cfhw: Tensor) -> Tensor:
        """``[C,F,H,W]`` → ``[C_lat,F,h,w]``."""
        c, f, h, w = rgb_cfhw.shape
        flat = rgb_cfhw.permute(1, 0, 2, 3).reshape(f, c, h, w)
        z = self.encode(flat)
        return z.permute(1, 0, 2, 3)

    def decode_video(self, z_cfhw: Tensor, *, out_h: int, out_w: int) -> Tensor:
        c, f, h, w = z_cfhw.shape
        flat = z_cfhw.permute(1, 0, 2, 3).reshape(f, c, h, w)
        rgb = self.decode(flat, out_hw=(out_h, out_w))
        return rgb.permute(1, 0, 2, 3).clamp(0.0, 1.0)
