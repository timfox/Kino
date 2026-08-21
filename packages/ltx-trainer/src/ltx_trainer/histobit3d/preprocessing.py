"""BIT and fluorescence preprocessing (Sec. 2.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def background_subtract(bit_slice: Tensor, *, sigma: float = 30.0) -> Tensor:
    """Background-subtract BIT with Gaussian σ=30 (Sec. 2.1)."""
    if bit_slice.dim() == 2:
        x = bit_slice.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif bit_slice.dim() == 3:
        x = bit_slice.unsqueeze(0)
        squeeze = False
    else:
        x = bit_slice
        squeeze = False
    # approximate large-σ blur via avg pool kernel sized from sigma
    k = max(3, int(sigma) | 1)
    pad = k // 2
    bg = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
    out = (x - bg).clamp(min=0.0)
    if squeeze:
        return out.squeeze(0).squeeze(0)
    if bit_slice.dim() == 3:
        return out.squeeze(0)
    return out


def bit_three_channel_stack(bit_slice: Tensor) -> Tensor:
    """Network input: (original, inverted, original) — Sec. 2.1."""
    if bit_slice.dim() == 2:
        orig = bit_slice
    else:
        orig = bit_slice[0] if bit_slice.shape[0] == 1 else bit_slice.mean(dim=0)
    inv = 1.0 - orig if orig.max() <= 1.0 else (orig.max() - orig)
    return torch.stack([orig, inv, orig], dim=0)


def scale_to_uint8(x: Tensor) -> Tensor:
    """Scale to 8-bit range [0, 255] for training."""
    x = x.float()
    xmin, xmax = x.min(), x.max()
    if (xmax - xmin) < 1e-8:
        return torch.zeros_like(x)
    return (255.0 * (x - xmin) / (xmax - xmin)).clamp(0, 255)
