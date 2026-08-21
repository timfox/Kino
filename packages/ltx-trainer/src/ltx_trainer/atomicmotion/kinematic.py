"""Temporal-Kinematic Block and Kinematic Attention (Sec. 3.3, Eq. 6–9)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.atomicmotion.config import AtomicMotionConfig
from ltx_trainer.atomicmotion.partitions import PARTITION_NAMES, PartitionName


def star_topology_weights(num_parts: int = 5) -> Tensor:
    """Initialize torso-rooted star: torso self=1, limbs from torso=0.1."""
    w = torch.eye(num_parts)
    w[0, 1:] = 0.1
    w[1:, 0] = 0.1
    return w


class TemporalAttention(nn.Module):
    """Per-partition temporal MSA with symmetric limb weight sharing (Eq. 6)."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.torso = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.limb = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, parts: dict[PartitionName, Tensor]) -> dict[PartitionName, Tensor]:
        out: dict[PartitionName, Tensor] = {}
        for name, feat in parts.items():
            x = feat.unsqueeze(0) if feat.dim() == 2 else feat
            if name == "torso":
                y, _ = self.torso(x, x, x)
            else:
                y, _ = self.limb(x, x, x)
            out[name] = y.squeeze(0) if feat.dim() == 2 else y
        return out


class KinematicAttention(nn.Module):
    """Dual-branch: dynamic S-MSA + static topology (Eq. 7–9)."""

    def __init__(self, dim: int, num_heads: int = 8, num_parts: int = 5):
        super().__init__()
        self.spatial = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.w_topo = nn.Parameter(star_topology_weights(num_parts))
        self.gate_attn = nn.Parameter(torch.full((num_parts, dim), 2.0))
        self.gate_topo = nn.Parameter(torch.full((num_parts, dim), -2.0))
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _aggregate_topo(self, h_stack: Tensor, w_topo: Tensor) -> Tensor:
        """Graph aggregation along partition dimension: [B, P, D]."""
        w = torch.tanh(w_topo)
        return torch.einsum("pq,bqd->bpd", w, h_stack)

    def forward(self, parts: dict[PartitionName, Tensor]) -> dict[PartitionName, Tensor]:
        names = list(PARTITION_NAMES)
        sample = parts[names[0]]
        if sample.dim() == 2:
            # [T, D] -> [1, P, D]
            stacked = torch.stack([parts[n] for n in names], dim=0).unsqueeze(0)
            layout = "t"
        elif sample.dim() == 3 and sample.shape[0] != len(names):
            # [B, T, D] -> [B*T, P, D]
            stacked = torch.stack([parts[n] for n in names], dim=2)
            b, t, p, d = stacked.shape
            stacked = stacked.reshape(b * t, p, d)
            layout = "bt"
            meta = (b, t)
        else:
            # [B, D] treated as [B, P, D] after stack on dim=1
            stacked = torch.stack([parts[n] for n in names], dim=1)
            layout = "b"
            meta = ()

        h = stacked
        attn_out, _ = self.spatial(h, h, h)
        topo_out = self._aggregate_topo(h, self.w_topo)
        ga = torch.sigmoid(self.gate_attn).unsqueeze(0)
        gt = torch.sigmoid(self.gate_topo).unsqueeze(0)
        z = self.norm1(h + ga * attn_out + gt * topo_out)
        z = self.norm2(z + self.ffn(z))

        out_parts: dict[PartitionName, Tensor] = {}
        if layout == "t":
            for i, name in enumerate(names):
                out_parts[name] = z[0, i, :]
        elif layout == "bt":
            b, t = meta
            z = z.reshape(b, t, len(names), -1)
            for i, name in enumerate(names):
                out_parts[name] = z[:, :, i, :]
        else:
            for i, name in enumerate(names):
                out_parts[name] = z[:, i, :]
        return out_parts


class TKBlock(nn.Module):
    """One Temporal-Kinematic block."""

    def __init__(self, cfg: AtomicMotionConfig | None = None):
        super().__init__()
        cfg = cfg or AtomicMotionConfig()
        self.temporal = TemporalAttention(cfg.embed_dim, cfg.num_heads)
        self.kinematic = KinematicAttention(cfg.embed_dim, cfg.num_heads)

    def forward(self, parts: dict[PartitionName, Tensor]) -> dict[PartitionName, Tensor]:
        h = self.temporal(parts)
        return self.kinematic(h)
