"""Anchor-Guided Injector (Eq. 1–2, Sec. 3.3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_embed(x: Tensor, dim: int) -> Tensor:
    """Scalar(s) -> sinusoidal features (B, dim)."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / max(half - 1, 1)
    )
    args = x * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)
    return emb


class FeatureRefinement(nn.Module):
    """1×1 then 3×3 conv to align anchor features with DiT channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, anchor: Tensor) -> Tensor:
        return self.net(anchor)


class AnchorGuidedInjector(nn.Module):
    """Dynamic gating: z_out = z + (1 + α) · z'_A (Eq. 2)."""

    def __init__(self, channels: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.refine = FeatureRefinement(channels)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.embed_dim = embed_dim

    def forward(self, z: Tensor, anchor: Tensor, timestep: Tensor, step_size: Tensor) -> Tensor:
        z_a = self.refine(anchor)
        t_emb = sinusoidal_embed(timestep.float(), self.embed_dim)
        dt_emb = sinusoidal_embed(step_size.float(), self.embed_dim)
        alpha = torch.tanh(self.gate_mlp(t_emb + dt_emb))
        gain = 1.0 + alpha.view(-1, 1, 1, 1)
        return z + gain * z_a
