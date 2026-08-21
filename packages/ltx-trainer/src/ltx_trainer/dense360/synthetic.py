"""Synthetic ERP + entity mask for CPU smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.dense360.config import Dense360Config


def synthetic_erp(cfg: Dense360Config) -> Tensor:
    """ERP RGB [1,3,H,W]."""
    h, w = cfg.height, cfg.width
    yy = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    base = (0.4 + 0.25 * torch.cos(xx * 6.28) * torch.sin(yy * 3.14)).clamp(0, 1)
    return base.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)


def synthetic_entity_mask(cfg: Dense360Config) -> Tensor:
    """Binary entity mask [1,1,H,W] near equator center."""
    h, w = cfg.height, cfg.width
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 1.0
    return mask
