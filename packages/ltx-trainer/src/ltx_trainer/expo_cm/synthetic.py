"""Synthetic LDR/HDR pairs for ExpoCM training."""

from __future__ import annotations

import torch
from torch import Tensor


def synthesize_hdr_scene(h: int = 256, w: int = 256) -> Tensor:
    hdr = torch.rand(3, h, w) * 0.5 + 0.02
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    spot = torch.exp(-(((yy.float() - h * 0.3) ** 2 + (xx.float() - w * 0.6) ** 2) / (0.06 * min(h, w) ** 2)))
    hdr = hdr + spot.unsqueeze(0) * torch.rand(1).item() * 1.5
    return hdr.clamp(min=0.0)


def ldr_from_hdr(hdr: Tensor, *, ev: float = 0.0, gamma: float = 2.2) -> Tensor:
    return (hdr * (2.0 ** ev)).clamp(0.0, 1.0).pow(1.0 / gamma)


def synthesize_pair(h: int = 128, w: int = 128) -> tuple[Tensor, Tensor]:
    hdr = synthesize_hdr_scene(h, w)
    peak = hdr.amax().clamp(min=1e-6)
    hdr_n = hdr / peak
    ldr = ldr_from_hdr(hdr_n, ev=0.0)
    return hdr_n, ldr
