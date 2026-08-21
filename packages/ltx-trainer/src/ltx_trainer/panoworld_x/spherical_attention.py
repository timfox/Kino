"""Sphere-Aware Attention mask and forward (Eq. 6–7)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.panoworld_x.erp_geometry import patch_spherical_coords, spherical_distance_haversine


def sphere_attention_mask(
    theta: Tensor,
    phi: Tensor,
    threshold: float,
) -> Tensor:
    """
    Eq. (6): M(p1, p2) = 1 if d_spherical ≤ τ else 0 (as additive mask: 0 / -inf).
    """
    n = theta.numel()
    t1 = theta.unsqueeze(1).expand(n, n)
    t2 = theta.unsqueeze(0).expand(n, n)
    p1 = phi.unsqueeze(1).expand(n, n)
    p2 = phi.unsqueeze(0).expand(n, n)
    dist = spherical_distance_haversine(t1, p1, t2, p2)
    allowed = dist <= threshold
    mask = torch.zeros(n, n, device=theta.device, dtype=theta.dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


def sphere_aware_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
) -> Tensor:
    """Eq. (7): softmax(QK^T/√d + M) V."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d) + mask
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


class SphereAwareAttention(nn.Module):
    """Parallel Sphere-Attn branch with zero-init output projection."""

    def __init__(self, dim: int, threshold: float, num_heads: int = 4) -> None:
        super().__init__()
        self.threshold = threshold
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        b, n, c = tokens.shape
        qkv = self.qkv(tokens).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        head_mask = mask.unsqueeze(0).unsqueeze(0).expand(b, self.num_heads, -1, -1)
        out = sphere_aware_attention(q, k, v, head_mask)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)
