"""Synthetic HDR/SDR pairs for training smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.lumivid.degrade import tone_map_reinhard


def synthesize_hdr_scene(h: int = 64, w: int = 64, *, device: torch.device | None = None) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device),
        torch.linspace(0, 1, w, device=device),
        indexing="ij",
    )
    base = 0.2 + 0.5 * yy
    highlight = torch.exp(-((xx - 0.75) ** 2 + (yy - 0.25) ** 2) / 0.02) * 8.0
    shadow = torch.exp(-((xx - 0.2) ** 2 + (yy - 0.8) ** 2) / 0.03) * 0.02
    rgb = torch.stack([base + highlight, base * 0.9 + highlight * 0.8, base * 0.7 + highlight * 0.6], dim=0)
    return (rgb + shadow).clamp(min=0.0)


def synthesize_sdr_from_hdr(hdr: Tensor) -> Tensor:
    return tone_map_reinhard(hdr)
