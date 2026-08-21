"""Synthetic ERP panoramas for CPU stub."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.par.config import PARConfig


def synthetic_erp(cfg: PARConfig | None = None, *, seed: int = 0) -> Tensor:
    cfg = cfg or PARConfig()
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.height, cfg.width
    u = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
    v = torch.linspace(0, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
    rgb = torch.cat([u, v, 0.5 * (u + v)], dim=1)
    return (rgb + 0.03 * torch.randn(1, 3, h, w, generator=g)).clamp(0, 1)


def synthetic_text(cfg: PARConfig | None = None) -> Tensor:
    cfg = cfg or PARConfig()
    return torch.randn(1, cfg.text_dim)
