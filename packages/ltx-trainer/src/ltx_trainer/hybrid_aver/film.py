"""FiLM-based audio conditioning (Eqs. 4–5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FiLMAudioConditioner(nn.Module):
    """γ, β from visual segment; â = (1+γ)⊙a + β."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

    def forward(self, audio: Tensor, visual: Tensor) -> Tensor:
        # audio, visual: (B, T, D)
        gamma = self.gamma(visual)
        beta = self.beta(visual)
        return (1.0 + gamma) * audio + beta
