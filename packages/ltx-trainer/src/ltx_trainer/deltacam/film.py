"""Feature-wise Linear Modulation for camera intrinsics (Eq. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DeltaFiLM(nn.Module):
    """Predict γ, β from Δθ and apply FiLM(h, Δθ) = γ ⊙ h + β."""

    def __init__(self, delta_dim: int, feature_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(delta_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * feature_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: Tensor, delta: Tensor) -> Tensor:
        """``h``: ``[B, C, …]`` or ``[C, …]``; ``delta``: ``[B, D]`` or ``[D]``."""
        if delta.dim() == 1:
            delta = delta.unsqueeze(0)
        if h.dim() == 3:
            # [C, H, W]
            gamma_beta = self.mlp(delta)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = gamma.squeeze(0).view(-1, 1, 1)
            beta = beta.squeeze(0).view(-1, 1, 1)
            return gamma * h + beta
        if h.dim() == 4:
            # [B, C, H, W]
            gamma_beta = self.mlp(delta)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            return gamma.view(-1, h.shape[1], 1, 1) * h + beta.view(-1, h.shape[1], 1, 1)
        raise ValueError(f"Unsupported feature rank {h.dim()}")


def film_modulate(h: Tensor, delta: Tensor, film: DeltaFiLM) -> Tensor:
    return film(h, delta)


def cascade_film(
    h: Tensor,
    deltas: list[Tensor],
    films: list[DeltaFiLM],
) -> Tensor:
    """Apply a sequence of FiLM layers (optical → sensory → ISP cascades)."""
    out = h
    for d, layer in zip(deltas, films, strict=True):
        out = layer(out, d)
    return out
