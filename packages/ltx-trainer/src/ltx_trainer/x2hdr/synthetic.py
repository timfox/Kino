"""Synthetic HDR/SDR pairs for X2HDR smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.lumivid.degrade import tone_map_reinhard


def synthesize_hdr_scene(h: int = 64, w: int = 64) -> Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = 0.15 + 0.6 * yy
    spot = torch.exp(-((xx - 0.7) ** 2 + (yy - 0.3) ** 2) / 0.015) * 12.0
    rgb = torch.stack([base + spot, base * 0.95 + spot * 0.9, base * 0.85 + spot * 0.8], dim=0)
    return rgb.clamp(min=0.0)


def synthesize_sdr_from_hdr(hdr: Tensor) -> Tensor:
    return tone_map_reinhard(hdr)
