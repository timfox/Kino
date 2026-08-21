"""MultiDiffusion latent fusion stub (Sec. 2.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def cyclic_pad_mask(mask: Tensor) -> Tensor:
    """Extend mask edges for cylindrical stitch."""
    return torch.cat([mask[..., -8:], mask, mask[..., :8]], dim=-1)


def merge_latents(paths: list[Tensor], masks: list[Tensor]) -> Tensor:
    """Average denoising predictions over overlapping mask regions."""
    if len(paths) == 1:
        return paths[0]
    acc = torch.zeros_like(paths[0])
    wsum = torch.zeros_like(paths[0][:, :1])
    for lat, m in zip(paths, masks):
        m = m.to(lat.dtype)
        if m.shape[-2:] != lat.shape[-2:]:
            m = torch.nn.functional.interpolate(m, size=lat.shape[-2:], mode="nearest")
        acc = acc + lat * m
        wsum = wsum + m
    return acc / wsum.clamp_min(1e-6)


def bootstrap_mask_focus(step: int, bootstrap: int, mask: Tensor) -> Tensor:
    """Early steps attend only to mask regions (Sec. 2.3)."""
    if step >= bootstrap:
        return torch.ones_like(mask)
    return mask
