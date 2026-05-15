"""Scalar EV → conditioning vector (Fourier features + MLP), LatentHDR-style."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EVFourierFeatures(nn.Module):
    """``2 * num_bands`` sinusoids of scalar EV (log₂ exposure offset)."""

    def __init__(self, num_bands: int = 32, max_log_freq: float = 9.0) -> None:
        super().__init__()
        self.num_bands = num_bands
        freqs = 2.0 ** torch.linspace(0.0, max_log_freq, num_bands)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, ev: torch.Tensor) -> torch.Tensor:
        """``ev`` shape ``[B]`` → ``[B, 2 * num_bands]``."""
        x = ev.float().unsqueeze(-1) * self.freqs * (2.0 * math.pi)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class EVConditionMLP(nn.Module):
    """Fourier features of EV followed by a small MLP (``φ(e)`` in LatentHDR)."""

    def __init__(self, num_bands: int = 32, hidden: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.fourier = EVFourierFeatures(num_bands=num_bands)
        in_dim = 2 * num_bands
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.SiLU(),
        )

    def forward(self, ev: torch.Tensor) -> torch.Tensor:
        return self.net(self.fourier(ev))
