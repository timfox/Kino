"""Physically-Guided Adaptation — gated LoRA on V projection (Sec. 4.3, Eq. 9–12)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PGALoRA(nn.Module):
    """Gated low-rank residual for attention value projection."""

    def __init__(self, dim: int, rank: int = 8, phys_channels: int = 32) -> None:
        super().__init__()
        self.av = nn.Linear(dim, rank, bias=False)
        self.bv = nn.Linear(rank, dim, bias=False)
        self.pv = nn.Conv2d(phys_channels + 1, dim, 1)
        self.wr = nn.Linear(8, dim)
        self.num_heads = 4
        self.head_dim = dim // self.num_heads

    def forward(
        self,
        v: Tensor,
        t_phys: Tensor,
        g_global: Tensor,
        spectral: Tensor,
        mod: dict[str, Tensor],
    ) -> Tensor:
        """
        Args:
            v: value tokens [B, N, D]
            t_phys: [B, C, H, W]
            g_global: [B, C]
            spectral: [B, K]
        """
        b, n, d = v.shape
        r_base = self.bv(self.av(v))
        h, w = t_phys.shape[-2:]
        g_scalar = g_global.mean(dim=1, keepdim=True)
        g_exp = g_scalar.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h, w)
        phys_stack = torch.cat([t_phys, g_exp], dim=1)
        gate_map = torch.sigmoid(self.pv(phys_stack))
        gate = gate_map.flatten(2).transpose(1, 2)
        if gate.shape[1] != n:
            side = int(n**0.5)
            if side * side != n:
                gate = F.adaptive_avg_pool1d(gate.transpose(1, 2), n).transpose(1, 2)
            else:
                gate = F.interpolate(gate_map, size=(side, side), mode="bilinear")
                gate = gate.flatten(2).transpose(1, 2)
        g_phys = gate.mean(dim=-1, keepdim=True)
        g_fft = torch.nn.functional.softplus(self.wr(spectral)).unsqueeze(1)
        alpha = mod["alpha_pga"].unsqueeze(1)
        beta = mod["beta_pga"].unsqueeze(1)
        n_spec = mod["n_spec"].unsqueeze(1)
        delta = (alpha * r_base + beta) * g_phys + n_spec * g_fft
        return v + delta
