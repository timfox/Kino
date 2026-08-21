"""Finite scalar quantization stub (FSQ levels [8,8,8,8,8])."""

from __future__ import annotations

import torch
from torch import Tensor


def fsq_quantize(z: Tensor, levels: tuple[int, ...] = (8, 8, 8, 8, 8)) -> Tensor:
    """Round each channel to FSQ bins; returns quantized latent (B, D)."""
    out = []
    for i, lv in enumerate(levels):
        if i >= z.shape[-1]:
            break
        ch = z[..., i]
        half = (lv - 1) / 2
        q = torch.round(torch.clamp(ch, -half, half))
        out.append(q)
    if not out:
        return z
    pad = z[..., len(out) :]
    return torch.cat([torch.stack(out, dim=-1), pad], dim=-1)


def codebook_size(levels: tuple[int, ...]) -> int:
    n = 1
    for lv in levels:
        n *= lv
    return n
