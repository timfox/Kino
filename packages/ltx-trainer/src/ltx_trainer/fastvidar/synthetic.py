"""Synthetic multi-fisheye ERP batch for smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.fastvidar.config import FastViDARConfig


def synthetic_erp_frames(cfg: FastViDARConfig | None = None) -> Tensor:
    cfg = cfg or FastViDARConfig()
    b, s, h, w = 1, cfg.num_frames, cfg.height, cfg.width
    frames = []
    for cam in range(s):
        v = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
        u = torch.linspace(0, 1, w).view(1, w).expand(h, w)
        phase = cam * 0.25
        r = v
        g = u
        bch = 0.5 + 0.3 * torch.sin(2 * 3.14159 * (u + phase))
        frames.append(torch.stack([r, g, bch], dim=0))
    return torch.stack(frames, dim=0).unsqueeze(0)


def synthetic_depth_gt(cfg: FastViDARConfig | None = None) -> Tensor:
    cfg = cfg or FastViDARConfig()
    h, w = cfg.height // 4, cfg.width // 4
    u = torch.linspace(-1, 1, w).view(1, w).expand(h, w)
    v = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
    return (2.0 + u.abs() + v.abs()).unsqueeze(0)
