"""Synthetic ERP views for CPU stub training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.erpgs.config import ErpGSConfig
from ltx_trainer.erpgs.erp_projection import distortion_weight_map


def synthetic_erp(cfg: ErpGSConfig | None = None, *, seed: int = 0) -> Tensor:
    cfg = cfg or ErpGSConfig()
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.height, cfg.width
    row = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    col = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    rgb = torch.stack([row, col, 0.5 * (row + col)], dim=0).unsqueeze(0)
    rgb = rgb + 0.05 * torch.randn(1, 3, h, w, generator=g)
    return rgb.clamp(0, 1)


def viewpoint_obstacle_mask(cfg: ErpGSConfig | None = None) -> Tensor:
    """Bottom-band mask for tripod / rig (viewpoint-dependent mask stub)."""
    cfg = cfg or ErpGSConfig()
    h, w = cfg.height, cfg.width
    mask = torch.ones(h, w)
    mask[int(0.85 * h) :, :] = 0.0
    return mask


def synthetic_batch(cfg: ErpGSConfig | None = None) -> tuple[Tensor, Tensor, Tensor]:
    cfg = cfg or ErpGSConfig()
    rgb = synthetic_erp(cfg)
    mask = viewpoint_obstacle_mask(cfg)
    weight = distortion_weight_map(cfg.height, cfg.width)
    return rgb, mask, weight
