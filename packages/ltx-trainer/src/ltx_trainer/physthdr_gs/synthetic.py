"""Synthetic multi-exposure LDR views for CPU training stub."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from ltx_trainer.physthdr_gs.config import EXPOSURE_TIMES, LDR_NE_INDICES, LDR_OE_INDICES


def _make_scene(size: int, seed: int) -> tuple[Tensor, Tensor]:
    """Return HDR radiance field and reflectance map [3,H,W]."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    hr = torch.rand(3, size, size, generator=g) * 0.4 + 0.2
    la = torch.rand(3, size, size, generator=g) * 0.6 + 0.4
    # Local highlight patch (luckycat-like nameplate)
    cx, cy = size // 3, size // 2
    r = size // 8
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r**2
    la[:, mask] *= 1.8
    hdr = (hr * la).clamp(0.0, 4.0)
    return hdr, hr


def tone_map_mu_law(hdr: Tensor, mu: float = 5000.0) -> Tensor:
    return torch.log1p(mu * hdr.clamp(min=0)) / torch.log1p(torch.tensor(mu))


def ldr_from_hdr(hdr: Tensor, exposure: float) -> Tensor:
    scaled = (hdr * exposure).clamp(0.0, 4.0)
    return tone_map_mu_law(scaled).clamp(0.0, 1.0)


def synthesize_view(
    size: int = 64,
    *,
    exposure: float = 1.0,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    hdr, _ = _make_scene(size, seed)
    ldr = ldr_from_hdr(hdr, exposure)
    return hdr.unsqueeze(0), ldr.unsqueeze(0)


def synthesize_multi_exposure(
    size: int = 64,
    *,
    seed: int = 0,
) -> tuple[Tensor, list[Tensor]]:
    hdr, _ = _make_scene(size, seed)
    hdr_b = hdr.unsqueeze(0)
    ldrs = [ldr_from_hdr(hdr, t).unsqueeze(0) for t in EXPOSURE_TIMES]
    return hdr_b, ldrs


def sample_training_pair(
    size: int = 64,
    *,
    exp3: bool = True,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, float]:
    """Return target LDR [1,3,H,W], HDR GT, exposure t."""
    s = seed if seed is not None else random.randint(0, 99999)
    hdr, ldrs = synthesize_multi_exposure(size, seed=s)
    if exp3:
        idx = random.choice(LDR_OE_INDICES)
    else:
        idx = LDR_OE_INDICES[1]
    t = EXPOSURE_TIMES[idx]
    return ldrs[idx], hdr, t
