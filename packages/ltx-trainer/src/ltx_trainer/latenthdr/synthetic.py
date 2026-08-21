"""Synthetic HDR scenes for LatentHDR training."""

from __future__ import annotations

import torch
from torch import Tensor


def synthesize_hdr_scene(h: int = 128, w: int = 128) -> Tensor:
    """Scene-linear HDR ``[3,H,W]`` with mixed bright/dark regions."""
    hdr = torch.rand(3, h, w) * 0.6 + 0.02
    spot = torch.zeros(1, h, w)
    cy, cx = h // 3, w // 2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    spot[0] = torch.exp(-(((yy - cy).float() ** 2 + (xx - cx).float() ** 2) / (0.08 * min(h, w) ** 2)))
    hdr = hdr + spot * torch.rand(1).item() * 2.0
    return hdr.clamp(min=0.0)


def synthesize_ldr_from_hdr(hdr: Tensor, *, ev: float = 0.0, gamma: float = 2.2) -> Tensor:
    scaled = (hdr * (2.0 ** ev)).clamp(0.0, 1.0)
    return scaled.pow(1.0 / gamma)
