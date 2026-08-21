"""E2S / S2E panorama projection stubs (Eq. 1–2, supp. Eq. 6)."""

from __future__ import annotations

import torch
from torch import Tensor


def erp_to_sphere_uv(x: Tensor, y: Tensor, z: Tensor, h: int, w: int) -> tuple[Tensor, Tensor]:
    """Map triangle center (x,y,z) to ERP (u,v) for E2S bilinear sample."""
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.sqrt(x * x + y * y).clamp_min(1e-6))
    u = (1 + lon / torch.pi) * w / 2
    v = (0.5 + lat / torch.pi) * h
    return u, v


def e2s_sample(erp_feat: Tensor, n_tri: int) -> Tensor:
    """Sample ERP features onto n_tri mesh vertices (stub grid)."""
    b, c, h, w = erp_feat.shape
    ys = torch.linspace(0, h - 1, n_tri, device=erp_feat.device)
    idx = ys.long().clamp(0, h - 1)
    row = erp_feat[:, :, idx, :].mean(dim=-1)
    return row.transpose(1, 2)
