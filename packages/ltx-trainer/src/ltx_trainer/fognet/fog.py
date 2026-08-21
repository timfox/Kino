"""Fog degradation for synthetic FogAct pairs (ASM-inspired)."""

from __future__ import annotations

import random

import torch
from torch import Tensor


def apply_fog(clean: Tensor, *, intensity: str = "light") -> Tensor:
    """Reduce contrast and add haze; ``clean`` is ``(C,H,W)`` or ``(T,C,H,W)``."""
    beta = 0.35 if intensity == "light" else 0.65
    airlight = random.uniform(0.7, 0.95)
    if clean.dim() == 4:
        out = []
        for t in range(clean.shape[0]):
            out.append(_fog_frame(clean[t], beta=beta, airlight=airlight))
        return torch.stack(out, dim=0)
    return _fog_frame(clean, beta=beta, airlight=airlight)


def _fog_frame(frame: Tensor, *, beta: float, airlight: float) -> Tensor:
    h, w = frame.shape[-2:]
    depth = torch.linspace(0.2, 1.0, h).view(h, 1).expand(h, w)
    t = torch.exp(-beta * depth)
    foggy = frame * t + airlight * (1.0 - t)
    # slight blur via avg pool
    if random.random() < 0.5:
        foggy = torch.nn.functional.avg_pool2d(foggy.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    return foggy.clamp(0, 1)
