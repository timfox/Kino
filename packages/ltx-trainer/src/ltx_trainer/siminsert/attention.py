"""Regional Attention Clone and Sparse Attention Fusion (Sec. III-A–B)."""

from __future__ import annotations

import torch
from torch import Tensor


def _expand_mask(mask: Tensor, values: Tensor) -> Tensor:
    m = mask.to(dtype=values.dtype, device=values.device)
    if m.dim() == values.dim() - 1:
        m = m.unsqueeze(-1)
    while m.dim() < values.dim():
        m = m.unsqueeze(0 if m.dim() == 1 else -1)
    if m.shape[-1] != values.shape[-1] and m.shape[-1] == 1:
        return m.expand(*values.shape[:-1], 1)
    return m


def regional_attention_clone(
    v_vis: Tensor,
    v_rec_vis: Tensor,
    mask: Tensor,
) -> Tensor:
    """Inject reconstruction values into background tokens (Eq. 4).

    mask: 1 = edited (keep v_vis), 0 = background (use v_rec_vis).
    """
    m = _expand_mask(mask, v_vis)
    return m * v_vis + (1.0 - m) * v_rec_vis


def sparse_attention_fusion(
    v_vis: Tensor,
    v_rec_vis: Tensor,
    mask: Tensor,
    *,
    retention_p: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Stochastic blend in background region (Eq. 5)."""
    m = _expand_mask(mask, v_vis)
    bg = 1.0 - m
    r = torch.bernoulli(
        torch.full(v_vis.shape[:-1], retention_p, device=v_vis.device),
        generator=generator,
    ).unsqueeze(-1)
    fused_bg = r * v_vis + (1.0 - r) * v_rec_vis
    return m * v_vis + bg * fused_bg


def apply_value_guidance(
    v_vis: Tensor,
    v_rec_vis: Tensor,
    mask: Tensor,
    *,
    use_clone: bool = True,
    use_fusion: bool = True,
    retention_p: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Full value-matrix guidance for one self-attention layer."""
    if use_fusion:
        return sparse_attention_fusion(
            v_vis,
            v_rec_vis,
            mask,
            retention_p=retention_p,
            generator=generator,
        )
    if use_clone:
        return regional_attention_clone(v_vis, v_rec_vis, mask)
    return v_vis
