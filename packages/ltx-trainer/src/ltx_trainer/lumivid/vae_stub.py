"""Lightweight VAE surrogate for roundtrip / latent training smoke."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VaeStub(nn.Module):
    """Conv encoder/decoder mimicking frozen video VAE latent shape."""

    def __init__(self, latent_ch: int = 16) -> None:
        super().__init__()
        self.latent_ch = latent_ch
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_ch, 3, stride=2, padding=1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, vae_rgb: Tensor) -> Tensor:
        if vae_rgb.dim() == 3:
            vae_rgb = vae_rgb.unsqueeze(0)
        return self.enc(vae_rgb)

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)
