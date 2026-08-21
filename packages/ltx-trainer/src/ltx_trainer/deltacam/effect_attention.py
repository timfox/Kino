"""Effect-attention path stub: fuse frozen-VAE latent with camera cond (Fig. 7, stage 1).

Full DiT blocks are external; this module is a minimal residual gate so new parameters
start near identity (zero-init projection), matching the paper's ``add new attention`` story.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EffectAttentionBlock(nn.Module):
    """Per-frame spatial latent ``z`` plus low-dim cond (Δ + pooled pose summary)."""

    def __init__(self, latent_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, latent_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        """``z``: ``[B, C, H, W]``; ``cond``: ``[B, D]`` → residual update."""
        if z.dim() != 4:
            raise ValueError(f"z must be [B,C,H,W], got {tuple(z.shape)}")
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if cond.shape[0] != z.shape[0]:
            raise ValueError("batch mismatch z vs cond")
        b, c, _, _ = z.shape
        gamma = self.proj(cond).view(b, c, 1, 1)
        return z + gamma * z


def concat_delta_plucker_cond(delta_t: Tensor, plucker_hw6: Tensor) -> Tensor:
    """Build ``[B, D+6]`` from Δ intrinsics and mean-pooled Plücker map ``[H, W, 6]``."""
    if plucker_hw6.dim() != 3 or plucker_hw6.shape[-1] != 6:
        raise ValueError(f"plucker_hw6 must be [H,W,6], got {tuple(plucker_hw6.shape)}")
    if delta_t.dim() == 1:
        delta_t = delta_t.unsqueeze(0)
    pooled = plucker_hw6.reshape(-1, 6).mean(dim=0)  # [6]
    pool_b = pooled.unsqueeze(0).expand(delta_t.shape[0], -1)
    return torch.cat([delta_t, pool_b], dim=-1)
