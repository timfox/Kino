"""Scalar exposure value (EV) embedding for the LatentHDR head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ExposureValueEmbedding(nn.Module):
    """Sinusoidal embedding of log2 exposure indices (LatentHDR φ(e) stub → MLP)."""

    def __init__(self, embed_dim: int = 64, max_log_freq: float = 8.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        freqs = 2.0 ** torch.linspace(0.0, max_log_freq, embed_dim // 2)
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, ev: Tensor) -> Tensor:
        if ev.ndim == 0:
            ev = ev.reshape(1)
        x = ev.reshape(-1, 1).to(dtype=self.freqs.dtype, device=self.freqs.device)
        angles = x * self.freqs.unsqueeze(0) * math.pi * 2.0
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = torch.nn.functional.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        elif emb.shape[-1] > self.embed_dim:
            emb = emb[..., : self.embed_dim]
        return self.proj(emb.to(dtype=self.proj.weight.dtype))
