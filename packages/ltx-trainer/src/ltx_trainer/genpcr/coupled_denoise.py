"""Coupled conditional denoising for cross-view texture consistency (Sec. 3.3.2, Eq. 3–4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def couple_latents(x_p: Tensor, x_q: Tensor) -> Tensor:
    """Stack source/target latents along height: xPQ ∈ R^{2H×W×d}."""
    return torch.cat([x_p, x_q], dim=-2)


def decouple_latents(x_pq: Tensor) -> tuple[Tensor, Tensor]:
    h2 = x_pq.shape[-2] // 2
    return x_pq[..., :h2, :], x_pq[..., h2:, :]


class CoupledDenoiseStub(nn.Module):
    """UNet-style stub: joint denoise on coupled latent with coupled depth/range cond."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x_pq: Tensor, cond_pq: Tensor) -> Tensor:
        return self.net(torch.cat([x_pq, cond_pq], dim=1))
