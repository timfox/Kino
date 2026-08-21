"""Circular Latent Encoding — Sec. 3.3, Eq. (3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.anything360.config import CLE_PAD_FRAC


def circular_pad_erp(panorama: Tensor, *, pad_frac: float = CLE_PAD_FRAC) -> Tensor:
    """Concat([Y[-w′:], Y, Y[:w′]]) along width for seam-free VAE encode."""
    if panorama.dim() != 4:
        raise ValueError("panorama must be B×C×H×W")
    w_pad = max(int(panorama.shape[-1] * pad_frac), 1)
    left = panorama[..., -w_pad:]
    right = panorama[..., :w_pad]
    return torch.cat([left, panorama, right], dim=-1)


def crop_circular_latent(latent: Tensor, *, pad_frac: float = CLE_PAD_FRAC) -> Tensor:
    """Drop latent columns corresponding to circular padding."""
    w_pad = max(int(latent.shape[-1] * pad_frac / (1 + 2 * pad_frac)), 1)
    return latent[..., w_pad:-w_pad]


class StubVAEEncoder(nn.Module):
    """Placeholder conv VAE encoder for CLE smoke tests."""

    def __init__(self, in_ch: int = 3, latent_ch: int = 16) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_ch, 3, padding=1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, 3, padding=1),
        )

    def forward(self, x: Tensor, *, circular: bool = True) -> Tensor:
        if circular:
            x = circular_pad_erp(x)
            z = self.enc(x)
            return crop_circular_latent(z)
        return self.enc(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)


def discontinuity_score(latent: Tensor) -> float:
    """DS proxy: mean |z[:,:,-1] - z[:,:,0]| across batch and channels."""
    if latent.dim() != 4:
        return 0.0
    diff = (latent[..., -1] - latent[..., 0]).abs().mean()
    return float(diff.item())
