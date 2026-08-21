"""Synthetic multi-exposure brackets for LF-Diff smoke."""

from __future__ import annotations

import torch
from torch import Tensor


def synthesize_hdr_scene(h: int = 64, w: int = 64) -> Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = 0.3 + 0.4 * yy
    sun = torch.exp(-((xx - 0.65) ** 2 + (yy - 0.35) ** 2) / 0.02) * 15.0
    rgb = torch.stack([base + sun, base * 0.9 + sun * 0.85, base * 0.8 + sun * 0.75], dim=0)
    return rgb.clamp(min=0.0)


def synthesize_exposure_bracket(hdr: Tensor, ev_steps: tuple[float, ...] = (-2.0, 0.0, 2.0)) -> list[Tensor]:
    return [(hdr * (2.0**ev)).clamp(0.0, 1.0) for ev in ev_steps]
