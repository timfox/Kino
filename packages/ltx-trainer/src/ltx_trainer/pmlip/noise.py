"""Learned functional perturbation modules (Sec. 3.1, Eq. 1–2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NoiseGenerator(nn.Module):
    """Sample z ~ N(0, Wz Wz^T) with shared graph-level noise."""

    def __init__(self, noise_dim: int = 32) -> None:
        super().__init__()
        self.wz = nn.Linear(noise_dim, noise_dim, bias=False)
        nn.init.eye_(self.wz.weight)

    def sample(self, batch_size: int = 1, *, device: torch.device | None = None) -> Tensor:
        device = device or self.wz.weight.device
        eps = torch.randn(batch_size, self.wz.in_features, device=device)
        return self.wz(eps)


class PerturbedLinear(nn.Module):
    """First linear sub-layer with zero-init noise projection (Eq. 2)."""

    def __init__(self, in_features: int, out_features: int, noise_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.noise_proj = nn.Linear(noise_dim, out_features, bias=False)
        nn.init.zeros_(self.noise_proj.weight)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        # x: (N, in), z: (noise_dim,) broadcast to all nodes in graph
        noise = self.noise_proj(z.unsqueeze(0)).expand(x.shape[0], -1)
        return self.linear(x) + noise


class PerturbedMLP(nn.Module):
    """Two-layer MLP; noise injected only at s=1."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, noise_dim: int) -> None:
        super().__init__()
        self.fc1 = PerturbedLinear(in_dim, hidden, noise_dim)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        h = torch.relu(self.fc1(x, z))
        return self.fc2(h)
