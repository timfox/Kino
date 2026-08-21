"""LLM orchestrator packed sequence (Eq. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class OrchestratorConditionHead(nn.Module):
    """Map E_cond(t) = [x_a^t, t] -> frame-aligned audio condition c_a."""

    def __init__(self, audio_dim: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, audio_dim),
        )

    def forward(self, x_a_t: Tensor, t: Tensor) -> Tensor:
        """x_a_t: (B, T, Ca); t: (B,) -> c_a (B, T, Ca)."""
        b, tok, c = x_a_t.shape
        t_exp = t.view(b, 1, 1).expand(b, tok, 1)
        inp = torch.cat([x_a_t, t_exp], dim=-1)
        return self.net(inp)
