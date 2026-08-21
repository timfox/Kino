"""Per-Gaussian emission MLP (Sec. 4.1, Eq. 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def complex_sigmoid(z: Tensor) -> Tensor:
    """Apply sigmoid to real and imaginary parts separately."""
    if z.shape[-1] == 2:
        return torch.sigmoid(z)
    return torch.sigmoid(z[..., :2])


class EmissionMLP(nn.Module):
    """f_θ : R³ → R²; s_i = σ(f_θ(x_rx) − log d_i)."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        x_rx: Tensor,
        log_distance: Tensor,
    ) -> Tensor:
        """
        Returns complex emission as (real, imag) last dim, shape broadcastable with log_distance.
        """
        base = self.net(x_rx)
        if base.dim() == 1:
            base = base.unsqueeze(0)
        while base.dim() < log_distance.dim() + 1:
            base = base.unsqueeze(0)
        out = base - log_distance.unsqueeze(-1)
        return complex_sigmoid(out)
