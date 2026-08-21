"""Foreground–Background Alignment Module (FBAM, Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _Adapt1x1(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(32 if dim >= 32 else 1, dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FBAMLayer(nn.Module):
    """Self-attention on image tokens + cross-attention with frozen background tokens."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.sa = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ca = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_sa = nn.LayerNorm(dim)
        self.ln_ca = nn.LayerNorm(dim)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x_i: Tensor, x_b: Tensor) -> Tensor:
        h, _ = self.sa(self.ln_sa(x_i), self.ln_sa(x_i), self.ln_sa(x_i), need_weights=False)
        x = x_i + h
        h, _ = self.ca(self.ln_ca(x), x_b, x_b, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln_mlp(x))
        return x


class FBAM(nn.Module):
    """Stack of FBAM layers with shared background conditioning tokens."""

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8) -> None:
        super().__init__()
        self.adapt = _Adapt1x1(dim)
        self.layers = nn.ModuleList(FBAMLayer(dim, num_heads) for _ in range(num_layers))

    def forward(self, image_map: Tensor, background_map: Tensor) -> Tensor:
        """Maps ``[B,D,Hb,Wb]`` → aligned features ``[B,D,Hb,Wb]``."""
        xi = self.adapt(image_map).flatten(2).transpose(1, 2)
        xb = self.adapt(background_map).flatten(2).transpose(1, 2)
        xb0 = xb
        for layer in self.layers:
            xi = layer(xi, xb0)
        b, _, h, w = image_map.shape
        return xi.transpose(1, 2).reshape(b, -1, h, w)
