"""Timestep and layer adaptive modulation Ψ(t, ℓ) (Sec. 4.2, Eq. 8)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_embed(x: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    args = x.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimestepLayerModulator(nn.Module):
    """Produces α, β, n_spec, λ for PGA, PCM, spectral gating, coupler."""

    def __init__(self, dim: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        self.layer_embed = nn.Embedding(num_layers, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.head_pga = nn.Linear(dim, 2)
        self.head_pcm = nn.Linear(dim, 2)
        self.head_spec = nn.Linear(dim, 1)
        self.head_lambda = nn.Linear(dim, 1)

    def forward(self, t: Tensor, layer_idx: int) -> dict[str, Tensor]:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = sinusoidal_embed(t, self.layer_embed.embedding_dim)
        l_emb = self.layer_embed(torch.tensor([layer_idx], device=t.device)).expand(t.shape[0], -1)
        h = self.mlp(torch.cat([t_emb, l_emb], dim=-1))
        alpha_pga, beta_pga = self.head_pga(h).chunk(2, dim=-1)
        alpha_pcm, beta_pcm = self.head_pcm(h).chunk(2, dim=-1)
        n_spec = torch.sigmoid(self.head_spec(h))
        lam = torch.sigmoid(self.head_lambda(h))
        return {
            "alpha_pga": alpha_pga,
            "beta_pga": beta_pga,
            "alpha_pcm": alpha_pcm,
            "beta_pcm": beta_pcm,
            "n_spec": n_spec,
            "lambda_t": lam,
        }
