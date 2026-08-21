"""Latent simulation slots and thought-video pooling (Sec. 3.1, 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.torm.config import TORMConfig


def pool_thought_video_targets(
    thought_tokens: Tensor,
    num_slots: int,
) -> Tensor:
    """Adaptive 1D average pool over token dim: (M, d) → (K, d)."""
    if thought_tokens.dim() == 2:
        h = thought_tokens.transpose(0, 1).unsqueeze(0)
    else:
        h = thought_tokens.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(h, num_slots)
    return pooled.squeeze(0).transpose(0, 1)


def latent_alignment_loss(z: Tensor, g: Tensor) -> Tensor:
    """Eq. (3): L_latent = (1/K) Σ ||z_i - g_i||²."""
    return F.mse_loss(z, g)


class LatentRolloutStub(nn.Module):
    """Bounded K-step latent rollout (hidden states fed back as embeddings)."""

    def __init__(self, cfg: TORMConfig) -> None:
        super().__init__()
        k, d = cfg.num_latent_slots, cfg.hidden_dim
        self.cell = nn.GRUCell(d, d)
        self.proj = nn.Linear(d, d)
        self.slots = nn.Parameter(torch.randn(k, d) * 0.02)

    def forward(self, context: Tensor) -> Tensor:
        """``context`` (B, d) pooled video-question state → (B, K, d) latent trajectory."""
        b, d = context.shape
        k = self.slots.shape[0]
        h = context
        states: list[Tensor] = []
        for i in range(k):
            h = self.cell(self.proj(h), self.slots[i].unsqueeze(0).expand(b, -1))
            states.append(h)
        return torch.stack(states, dim=1)
