"""Spherical epipolar attention (Sec. 3.3.3, Eq. 10)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def spherical_epipolar_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Eq. (10): softmax(q k^T / sqrt(d) ⊙ M) v.

    query: [H*W, C]; key/value: [N*H*W, C]; mask: [H*W, N*H*W].
    """
    d = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d)
    scores = scores.masked_fill(mask <= 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, value)


class SphericalEpipolarAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, q_frame: Tensor, kv_frames: Tensor, mask: Tensor) -> Tensor:
        """
        q_frame: [HW, C]; kv_frames: [N, HW, C]; mask: [HW, N*HW] additive.
        """
        n, hw, c = kv_frames.shape
        k = kv_frames.reshape(n * hw, c)
        v = k
        attn_mask = mask <= 0
        out, _ = self.attn(q_frame.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=attn_mask)
        return out.squeeze(0)
