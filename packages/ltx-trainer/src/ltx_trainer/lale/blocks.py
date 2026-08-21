"""ConvMixer and transformer blocks (§3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.lale.ops import RMSNorm, StarReLU


class ConvMixerBlock(nn.Module):
    """Depthwise spatial mix + pointwise channel mix."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm1 = RMSNorm(channels)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.act = StarReLU()
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.norm2 = RMSNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.dw(x)
        y = self.norm1(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.norm2(y)
        return x + y


class ConvMLP(nn.Module):
    """ConvMLP: linear expand → 3×3 depthwise → linear reduce (§3.3)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = StarReLU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: B, N, C
        b, n, c = x.shape
        h = w = int(n**0.5)
        y = self.fc1(x)
        y = y.transpose(1, 2).reshape(b, -1, h, w)
        y = self.dw(y)
        y = self.act(y)
        y = y.flatten(2).transpose(1, 2)
        return self.fc2(y)


class TransformerBlock(nn.Module):
    """MHSA + ConvMLP with RMSNorm (late stages)."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ConvMLP(dim)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        n1 = self.norm1(tokens)
        attn_out, _ = self.attn(n1, n1, n1)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)
