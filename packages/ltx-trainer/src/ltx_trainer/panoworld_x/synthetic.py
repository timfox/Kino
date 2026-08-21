"""Synthetic ERP frame + route for smoke tests."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.panoworld_x.config import PanoWorldXConfig
from ltx_trainer.panoworld_x.exploration_route import route_to_plucker_volume


def synthetic_erp_frame(cfg: PanoWorldXConfig | None = None) -> Tensor:
    cfg = cfg or PanoWorldXConfig()
    h, w = cfg.height, cfg.width
    v = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    u = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    r = v
    g = u
    b = 0.5 * (torch.sin(6 * u * 3.14159) + torch.cos(4 * v * 3.14159))
    return torch.stack([r, g, b], dim=0).unsqueeze(0)


def synthetic_route(cfg: PanoWorldXConfig | None = None) -> Tensor:
    cfg = cfg or PanoWorldXConfig()
    t = cfg.num_frames
    poses = torch.zeros(t, 6)
    poses[:, 0] = torch.linspace(0, 5, t)
    poses[:, 3] = torch.linspace(0, 0.5, t)
    plucker = route_to_plucker_volume(poses, cfg.height, cfg.width)
    return plucker.unsqueeze(0)
