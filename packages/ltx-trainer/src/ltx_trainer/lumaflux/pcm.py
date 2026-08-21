"""Perceptual Cross-Modulation — SigLIP FiLM (Sec. 4.4, Eq. 13–14)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SigLIPStub(nn.Module):
    """Frozen SigLIP image encoder stub (Eq. 7)."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 16, embed_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, sdr: Tensor) -> Tensor:
        return self.encoder(sdr).unsqueeze(1)


class PerceptualConnector(nn.Module):
    """Cperc: project SigLIP tokens to DiT width."""

    def __init__(self, in_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, t_perc: Tensor) -> Tensor:
        return self.proj(t_perc)


class PCMFiLM(nn.Module):
    """FiLM conditioning on layer-normalized hidden states (Eq. 14)."""

    def __init__(self, dim: int, perc_dim: int = 64) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Linear(perc_dim, dim * 2)

    def forward(self, h: Tensor, t_perc: Tensor, mod: dict[str, Tensor]) -> Tensor:
        """
        Args:
            h: [B, N, D] hidden activations
            t_perc: [B, Np, D] projected perceptual tokens
        """
        pooled = t_perc.mean(dim=1)
        gamma, zeta = self.mlp(pooled).chunk(2, dim=-1)
        alpha = mod["alpha_pcm"]
        beta = mod["beta_pcm"]
        gamma = alpha * gamma.unsqueeze(1) + beta.unsqueeze(1)
        zeta = alpha * zeta.unsqueeze(1) + beta.unsqueeze(1)
        return gamma * self.ln(h) + zeta
