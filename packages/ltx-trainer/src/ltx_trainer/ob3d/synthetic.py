"""Synthetic ERP RGB / depth / normal for stub pipelines."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.ob3d.config import OB3DConfig


def erp_uv_grid(h: int, w: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """Latitude θ and longitude φ for equirectangular pixels."""
    row = torch.arange(h, device=device, dtype=torch.float32)
    col = torch.arange(w, device=device, dtype=torch.float32)
    theta = (row + 0.5) * (math.pi / h) - math.pi / 2
    phi = (col + 0.5) * (2 * math.pi / w) - math.pi
    th = theta.view(h, 1).expand(h, w)
    ph = phi.view(1, w).expand(h, w)
    return th, ph


def synthetic_rgb(cfg: OB3DConfig | None = None) -> Tensor:
    cfg = cfg or OB3DConfig()
    h, w = cfg.height, cfg.width
    th, ph = erp_uv_grid(h, w)
    r = (torch.sin(3 * ph) * torch.cos(2 * th) + 1) * 0.5
    g = (torch.cos(ph) * 0.5 + 0.5)
    b = (th / (math.pi / 2) + 1) * 0.5
    rgb = torch.stack([r, g, b], dim=0).unsqueeze(0)
    return rgb.clamp(0, 1)


def synthetic_depth(cfg: OB3DConfig | None = None) -> Tensor:
    cfg = cfg or OB3DConfig()
    h, w = cfg.height, cfg.width
    th, ph = erp_uv_grid(h, w)
    depth = 3.0 + 2.0 * torch.cos(th) + 0.5 * torch.sin(2 * ph)
    depth = depth.clamp(0.5, cfg.max_depth)
    depth[..., : w // 8] = cfg.max_depth
    return depth.unsqueeze(0).unsqueeze(0)


def synthetic_normal(cfg: OB3DConfig | None = None) -> Tensor:
    cfg = cfg or OB3DConfig()
    h, w = cfg.height, cfg.width
    th, ph = erp_uv_grid(h, w)
    nx = torch.sin(ph) * torch.cos(th)
    ny = torch.sin(th)
    nz = torch.cos(ph) * torch.cos(th)
    n = torch.stack([nx, ny, nz], dim=0)
    n = n / n.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return n.unsqueeze(0)
