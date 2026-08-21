"""AVControl-style flow-matching DiT stub (Fig. 2–3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lumivid.lora import inject_lora


def _time_embed(t: Tensor, dim: int) -> Tensor:
    if t.ndim == 0:
        t = t.reshape(1)
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t.reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowDiTStub(nn.Module):
    """Predicts velocity field for rectified flow: ``v = z_tgt - noise``."""

    def __init__(self, *, latent_ch: int = 16, base_ch: int = 32, tdim: int = 64, lora_rank: int = 4) -> None:
        super().__init__()
        self.latent_ch = latent_ch
        self.t_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        c1, c2 = base_ch, base_ch * 2
        # AVControl: concat noisy target + reference condition
        self.in_conv = nn.Conv2d(latent_ch * 2, c1, 3, padding=1)
        self.down = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(c2, c2, 3, padding=1)
        self.up = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.out = nn.Conv2d(c1, latent_ch, 3, padding=1)
        self.tdim = tdim
        self.t_proj = nn.Linear(tdim, c2)
        self._lora = inject_lora(self, rank=lora_rank)

    def forward(self, z_t: Tensor, z_ref: Tensor, t: Tensor) -> Tensor:
        squeeze = z_t.dim() == 3
        if squeeze:
            z_t = z_t.unsqueeze(0)
            z_ref = z_ref.unsqueeze(0)
        b = z_t.shape[0]
        if t.ndim == 0:
            t = t.reshape(1).expand(b)
        temb = self.t_mlp(_time_embed(t, self.tdim))
        x = torch.cat([z_t, z_ref], dim=1)
        h = F.silu(self.in_conv(x))
        h = F.silu(self.down(h))
        scale = self.t_proj(temb).view(b, -1, 1, 1)
        h = F.silu(self.mid(h) * (1.0 + scale))
        h = F.silu(self.up(h))
        v = self.out(h)
        return v.squeeze(0) if squeeze else v

    def lora_parameters(self):
        for adapter in self._lora:
            yield from adapter.lora_a.parameters()
            yield from adapter.lora_b.parameters()
