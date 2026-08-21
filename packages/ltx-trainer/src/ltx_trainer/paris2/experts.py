"""Expert pool and routed velocity combination (Sec. 2–3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.paris2.config import Paris2Config
from ltx_trainer.paris2.flow_matching import ExpertVelocityStub


def top_k_mask(weights: Tensor, k: int) -> Tensor:
    """Keep top-k experts and renormalize to sum to 1."""
    if k >= weights.shape[-1]:
        return weights
    _, idx = weights.topk(k, dim=-1)
    masked = torch.zeros_like(weights)
    masked.scatter_(-1, idx, 1.0)
    return masked / masked.sum(dim=-1, keepdim=True).clamp(min=1e-6)


class ExpertPool(nn.Module):
    """Independent experts; no gradient sync during training."""

    def __init__(self, cfg: Paris2Config) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [ExpertVelocityStub(cfg.latent_channels) for _ in range(cfg.num_experts)]
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Weighted sum of expert velocity fields."""
        v = 0.0
        for i, expert in enumerate(self.experts):
            w = weights[:, i].view(-1, 1, 1, 1, 1)
            v = v + w * expert(x_t, t)
        return v

    def expert_velocity(self, expert_idx: int, x_t: Tensor, t: Tensor) -> Tensor:
        return self.experts[expert_idx](x_t, t)


def alternating_schedule_weights(
    step: int,
    num_experts: int = 2,
) -> Tensor:
    """Manual A→B switching schedule (Sec. 4.3 ablation)."""
    idx = step % num_experts
    w = torch.zeros(num_experts)
    w[idx] = 1.0
    return w
