"""Input-dependent κ, φ, ψ gates (NCP-inspired shared backbone; paper §3.1)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class OUGateHead(nn.Module):
    """Maps concatenated (q, k) features to OU parameters with positivity / tanh constraints."""

    def __init__(self, pair_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        h = hidden_dim or max(4 * pair_dim, 32)
        self.net = nn.Sequential(
            nn.Linear(pair_dim, h),
            nn.GELU(),
            nn.Linear(h, 3),
        )

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``u`` shape (..., pair_dim) → κ, φ, ψ with same leading shape."""
        raw = self.net(u)
        kappa = torch.nn.functional.softplus(raw[..., 0]) + 1e-4
        phi = torch.tanh(raw[..., 1])
        psi = torch.nn.functional.softplus(raw[..., 2]) + 1e-4
        return kappa, phi, psi
