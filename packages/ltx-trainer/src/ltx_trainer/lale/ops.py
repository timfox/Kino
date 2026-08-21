"""RMSNorm and StarReLU (§3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """Root mean square normalization (no mean centering)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) or (B, N, C)
        if x.dim() == 4:
            rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight.view(1, -1, 1, 1)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class StarReLU(nn.Module):
    """StarReLU: scale * ReLU(x)² + bias (MetaFormer-style)."""

    def __init__(self, scale: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * F.relu(x).pow(2) + self.bias
