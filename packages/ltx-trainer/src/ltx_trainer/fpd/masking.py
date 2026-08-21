"""Discrete diffusion masking (Eq. 1–2)."""

from __future__ import annotations

import torch
from torch import Tensor


def mask_probability(t: Tensor | float, gamma_max: float = 1.0) -> Tensor:
    """Monotonic noise schedule γ(t) for t ∈ [0, 1]."""
    t_t = torch.as_tensor(t, dtype=torch.float32)
    return gamma_max * t_t


def forward_mask_tokens(
    z: Tensor,
    t: float,
    *,
    mask_id: int,
    gamma_max: float = 1.0,
) -> Tensor:
    """Eq. (1): independently replace positions with [M] with probability γ(t)."""
    p_mask = float(mask_probability(t, gamma_max))
    z_t = z.clone()
    bern = torch.rand_like(z, dtype=torch.float32) < p_mask
    z_t[bern] = mask_id
    return z_t


def remask_fraction(z: Tensor, mask_id: int, r: float) -> Tensor:
    """Partial re-mask Mr(ẑ) at ratio r for teacher refinement (Eq. 5)."""
    z_r = z.clone()
    n = z.numel()
    k = max(1, int(n * r))
    idx = torch.randperm(n, device=z.device)[:k]
    z_r.view(-1)[idx] = mask_id
    return z_r
