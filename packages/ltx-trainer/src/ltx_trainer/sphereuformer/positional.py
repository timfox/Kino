"""Vertical global PE + 7×7 relative bias (Sec. 4.2–4.3, Fig. 6)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

F_MIN = 1.0
F_MAX = 10.0
PE_DIM = 16


def vertical_sinusoidal_pe(phi: Tensor, dim: int = PE_DIM) -> Tensor:
    """ϕ ∈ [0, π] → 2·dim features (Eq. 17 supplement)."""
    phi_hat = 2.0 * phi - math.pi
    i = torch.arange(dim, device=phi.device, dtype=phi.dtype)
    freq = F_MAX ** (i / max(dim - 1, 1))
    angles = phi_hat.unsqueeze(-1) * freq
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class VerticalGlobalPE(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * PE_DIM, out_dim)

    def forward(self, phi: Tensor) -> Tensor:
        return self.proj(vertical_sinusoidal_pe(phi))


class RelativePositionBias(nn.Module):
    """Learned 7×7 grid sampled by bilinear (∆θ, ∆ϕ) stub."""

    def __init__(self, grid: int = 7, heads: int = 4) -> None:
        super().__init__()
        self.grid = grid
        self.bias = nn.Parameter(torch.zeros(heads, grid, grid))

    def forward(self, delta_theta: Tensor, delta_phi: Tensor) -> Tensor:
        # delta: (B, N, K) normalized to [-1, 1]
        gx = ((delta_theta.clamp(-1, 1) + 1) * 0.5 * (self.grid - 1)).long().clamp(0, self.grid - 1)
        gy = ((delta_phi.clamp(-1, 1) + 1) * 0.5 * (self.grid - 1)).long().clamp(0, self.grid - 1)
        h = self.bias.shape[0]
        b, n, k = gx.shape
        flat = self.bias[:, gy.reshape(-1), gx.reshape(-1)].view(h, b, n, k).mean(dim=0)
        return flat  # (B, N, K)
