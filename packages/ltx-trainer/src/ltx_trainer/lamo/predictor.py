"""Micro motion-field predictor f_φ (Sec. 3.3, ~10M-param CNN reference)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel recalibration (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class FiLMResidualBlock(nn.Module):
    """Residual block with FiLM conditioning from prompt embedding."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.se = SEBlock(channels)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.norm1(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        h = torch.relu(self.conv1(h))
        h = self.norm2(h)
        h = torch.relu(self.conv2(h))
        h = self.se(h)
        return x + h


class MotionFieldPredictor(nn.Module):
    """f_φ : (z, c) ↦ Δτ z spatial field; zero-init output (Sec. 3.3)."""

    def __init__(
        self,
        in_channels: int,
        *,
        cond_dim: int = 64,
        hidden_channels: int = 32,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(hidden_channels, cond_dim) for _ in range(num_blocks)
        )
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        for block in self.blocks:
            nn.init.zeros_(block.film.weight)
            nn.init.zeros_(block.film.bias)

    def forward(self, z: Tensor, cond: Tensor | None = None) -> Tensor:
        """z: (B,C,H,W); cond: (B,cond_dim) or None for null embedding."""
        if cond is None:
            cond = self.null_cond.unsqueeze(0).expand(z.size(0), -1)
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h)
