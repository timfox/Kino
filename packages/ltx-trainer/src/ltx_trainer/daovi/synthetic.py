"""Synthetic ERP clips for CPU smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.daovi.config import DaoviConfig


def synthetic_erp_video(cfg: DaoviConfig) -> Tensor:
    """[1, T, 3, H, W] gradient ERP-like clip."""
    b, t, h, w = 1, cfg.num_frames, cfg.height, cfg.width
    yy = torch.linspace(0, 1, h).view(1, 1, h, 1).expand(b, t, h, w)
    xx = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(b, t, h, w)
    tt = torch.linspace(0, 1, t).view(1, t, 1, 1).expand(b, t, h, w)
    r = yy
    g = xx
    bl = tt.expand(b, t, h, w)
    return torch.stack([r, g, bl], dim=2)


def synthetic_masks(cfg: DaoviConfig) -> Tensor:
    """[1, T, 1, H, W] moving rectangular mask."""
    m = torch.zeros(1, cfg.num_frames, 1, cfg.height, cfg.width)
    for ti in range(cfg.num_frames):
        y0 = (ti * 3) % (cfg.height - 16)
        x0 = (ti * 5) % (cfg.width - 24)
        m[:, ti, :, y0 : y0 + 16, x0 : x0 + 24] = 1.0
    return m


def synthetic_flow(cfg: DaoviConfig) -> tuple[Tensor, Tensor]:
    """Forward/backward flow [1, T-1, 2, H, W]."""
    t = cfg.num_frames - 1
    fwd = torch.zeros(1, t, 2, cfg.height, cfg.width)
    bwd = torch.zeros_like(fwd)
    fwd[:, :, 0] = 1.0
    bwd[:, :, 0] = -1.0
    return fwd, bwd


def synthetic_depth(cfg: DaoviConfig) -> Tensor:
    """[1, T, 1, H, W] depth prior."""
    yy = torch.linspace(0.2, 1.0, cfg.height).view(1, 1, cfg.height, 1)
    return yy.expand(1, cfg.num_frames, 1, cfg.height, cfg.width).clone()
