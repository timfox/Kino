"""Synthetic LR ERP patches."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.mddn.config import MddnConfig


def synthetic_lr_erp(cfg: MddnConfig, *, batch_size: int = 1) -> Tensor:
    h, w = cfg.erp_height, cfg.erp_width
    rows = torch.linspace(0, 1, h).view(1, 1, h, 1).expand(batch_size, 1, h, w)
    cols = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(batch_size, 1, h, w)
    base = 0.35 * rows + 0.25 * cols + 0.15 * torch.sin(cols * 12.0)
    return base.clamp(0, 1).expand(batch_size, cfg.in_channels, h, w)
