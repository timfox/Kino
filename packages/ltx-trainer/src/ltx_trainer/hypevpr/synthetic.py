"""Toy perspective query + equirectangular database."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.hypevpr.config import HypeVPRConfig


def synthetic_query(cfg: HypeVPRConfig | None = None) -> Tensor:
    cfg = cfg or HypeVPRConfig()
    h = w = cfg.query_size
    y = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
    x = torch.linspace(-1, 1, w).view(1, w).expand(h, w)
    rgb = torch.stack([x, y, (x * y) * 0.5 + 0.5], dim=0)
    return rgb.unsqueeze(0)


def synthetic_panorama(cfg: HypeVPRConfig | None = None) -> Tensor:
    """Panorama W' = ratio × W with repeated / shifted query content."""
    cfg = cfg or HypeVPRConfig()
    h = cfg.query_size
    wp = h * cfg.pano_width_ratio
    base = synthetic_query(cfg)
    tiles = []
    for i in range(cfg.pano_width_ratio):
        shift = (i * h // 4) % h
        tiles.append(torch.roll(base.squeeze(0), shifts=shift, dims=-1))
    pano = torch.cat(tiles, dim=-1)
    if pano.shape[-1] > wp:
        pano = pano[..., :wp]
    elif pano.shape[-1] < wp:
        pano = torch.nn.functional.pad(pano, (0, wp - pano.shape[-1]))
    return pano.unsqueeze(0)
