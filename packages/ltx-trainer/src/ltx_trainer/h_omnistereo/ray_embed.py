"""Spherical ray positional encoding for ERP pixels."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class RayEmbedding(nn.Module):
    """Sine-cosine embedding of (θ, φ) ray directions for ERP."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = dim
        freqs = torch.exp(torch.linspace(0, math.log(1000), dim // 4))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, height: int, width: int, *, device: torch.device | None = None) -> Tensor:
        v = torch.arange(height, device=device, dtype=torch.float32)
        u = torch.arange(width, device=device, dtype=torch.float32)
        theta = (v + 0.5) / height * math.pi
        phi = (u + 0.5) / width * 2.0 * math.pi
        th, ph = torch.meshgrid(theta, phi, indexing="ij")
        rays = torch.stack([th, ph], dim=0).unsqueeze(0)  # [1,2,H,W]
        emb: list[Tensor] = []
        for i in range(rays.shape[1]):
            x = rays[:, i : i + 1]
            for f in self.freqs:
                emb.append(torch.sin(x * f))
                emb.append(torch.cos(x * f))
        out = torch.cat(emb, dim=1)
        if out.shape[1] > self.dim:
            return out[:, : self.dim]
        if out.shape[1] < self.dim:
            return torch.nn.functional.pad(out, (0, 0, 0, 0, 0, self.dim - out.shape[1]))
        return out
