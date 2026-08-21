"""Circular horizontal padding for VAE encode/decode (Sec. 3.4, Appendix A.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pad_horizontal_circular(x: Tensor, pad_w: int) -> Tensor:
    """Wrap azimuth (width) edges; zero-pad height only."""
    if pad_w <= 0:
        return x
    return F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")


class CircularConv2d(nn.Module):
    """Conv2d with circular padding on W (ERP azimuth) and zero pad on H."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = pad_horizontal_circular(x, self.pad)
        if self.pad > 0:
            x = F.pad(x, (0, 0, self.pad, self.pad))
        return self.conv(x)


class CircularVAEStub(nn.Module):
    """Frozen VAE surrogate: circular conv encoder/decoder for smoke tests."""

    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            CircularConv2d(3, 16),
            nn.GELU(),
            CircularConv2d(16, latent_channels),
        )
        self.decoder = nn.Sequential(
            CircularConv2d(latent_channels, 16),
            nn.GELU(),
            CircularConv2d(16, 3),
        )

    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)
