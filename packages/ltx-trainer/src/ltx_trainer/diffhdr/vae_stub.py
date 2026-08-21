"""Frozen video VAE stub (Wan-2.1-VAE, 4×8×8 compression)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VideoVAEStub(nn.Module):
    """Spatiotemporal VAE encode/decode stub; weights frozen."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, latent_dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 3, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Sigmoid(),
        )
        for p in self.parameters():
            p.requires_grad = False

    def encode(self, video: Tensor) -> Tensor:
        """video [B,T,C,H,W] → latent [B,C',T',H',W']."""
        if video.dim() == 4:
            video = video.unsqueeze(0)
        x = video.permute(0, 2, 1, 3, 4)
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder(z)
        return x.permute(0, 2, 1, 3, 4)
