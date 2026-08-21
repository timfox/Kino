"""STOccDiT: spatio-temporal occupancy diffusion transformer (Sec. 3.2)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.anyscene.config import AnySceneConfig


def causal_temporal_mask(t_frames: int, *, device: torch.device | None = None) -> Tensor:
    """Fig. 3b: noisy token at t attends only to clean tokens from past frames."""
    n = 2 * t_frames
    mask = torch.full((n, n), float("-inf"), device=device)
    for t in range(t_frames):
        noisy_idx = 2 * t + 1
        for s in range(t + 1):
            clean_idx = 2 * s
            mask[noisy_idx, clean_idx] = 0.0
        mask[noisy_idx, noisy_idx] = 0.0
    return mask


def flow_matching_noisy_latent(z: Tensor, tau: Tensor, noise: Tensor | None = None) -> Tensor:
    """Eq. supplement (3): z^τ = (1-τ) z + τ ε."""
    if noise is None:
        noise = torch.randn_like(z)
    tau = tau.view(-1, 1, 1)
    return (1.0 - tau) * z + tau * noise


def concat_bev_occ_tokens(bev: Tensor, occ: Tensor) -> Tensor:
    """Token concat conditioning (Tab. 4 ablation)."""
    return torch.cat([bev, occ], dim=1)


class STOccDiTBlock(nn.Module):
    """Spatial self-attn on concat tokens + causal temporal attn (stub)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.spatial = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.temporal = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.temporal_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: Tensor,
        *,
        temporal_mask: Tensor | None = None,
    ) -> Tensor:
        b, s, d = x.shape
        x_norm = F.layer_norm(x, (d,))
        attn, _ = self.spatial(x_norm, x_norm, x_norm)
        x = x + attn
        if temporal_mask is not None:
            x_t = x.transpose(0, 1)
            t_out, _ = self.temporal(x_t, x_t, x_t, attn_mask=temporal_mask)
            x = x + self.temporal_gate * t_out.transpose(0, 1)
        x = x + self.ff(F.layer_norm(x, (d,)))
        return x


class STOccDiT(nn.Module):
    """Lightweight STOccDiT for smoke tests (depth configurable)."""

    def __init__(self, cfg: AnySceneConfig, *, depth: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dit_hidden
        self.in_proj = nn.Linear(cfg.latent_channels, d)
        self.bev_proj = nn.Linear(cfg.class_embed_dim, d)
        n_blocks = depth if depth is not None else min(4, cfg.dit_depth)
        self.blocks = nn.ModuleList([STOccDiTBlock(d, cfg.dit_heads) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(d, cfg.latent_channels)

    def forward(
        self,
        z_noisy: Tensor,
        bev_tokens: Tensor,
        *,
        temporal_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict velocity on occupancy latent tokens."""
        occ = self.in_proj(z_noisy)
        bev = self.bev_proj(bev_tokens)
        x = concat_bev_occ_tokens(bev, occ)
        for blk in self.blocks:
            x = blk(x, temporal_mask=temporal_mask)
        return self.out_proj(x[:, bev.shape[1] :])


def flow_matching_loss(pred_v: Tensor, z: Tensor, noise: Tensor, tau: Tensor) -> Tensor:
    """MSE between predicted and target velocity ε - z."""
    target = noise - z
    w = 1.0 / (tau.clamp(min=1e-3))
    return (w.view(-1, 1, 1) * (pred_v - target).pow(2)).mean()
