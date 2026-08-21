"""T–V fused sparse attention mask (Sec. 3.2, Eq. 10)."""

from __future__ import annotations

import torch
from torch import Tensor


def tv_fused_mask(
    num_views: int,
    num_frames: int,
    *,
    source_view: int = 0,
) -> Tensor:
    """Binary mask M ∈ R^{N×N}, N = Nv · 2f · s (s=1 for smoke)."""
    f2 = 2 * num_frames
    n = num_views * f2
    v_idx = torch.arange(n) // f2
    t_idx = torch.arange(n) % f2
    vi = v_idx.unsqueeze(1)
    vj = v_idx.unsqueeze(0)
    ti = t_idx.unsqueeze(1)
    tj = t_idx.unsqueeze(0)
    same_view = vi == vj
    same_time = ti == tj
    cross_half = (ti < num_frames) & (tj == ti + num_frames) & (vj == source_view)
    m = (same_view | same_time | cross_half).float()
    return m


def masked_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    """Softmax(QK^T / sqrt(d) ⊙ M) V (Eq. 9)."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d**0.5)
    scores = scores.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
