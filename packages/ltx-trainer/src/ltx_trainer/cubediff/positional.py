"""Cube UV positional encoding (Eq. 1)."""

from __future__ import annotations

import torch
from torch import Tensor


def cube_uv_encoding(h: int, w: int, face_idx: int, device: torch.device | None = None) -> Tensor:
    """Per-pixel (u,v) on unit cube face, normalized to [0,1]."""
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # face-dependent rotation stub
    ang = torch.tensor(face_idx * (3.14159265 / 2), device=device, dtype=xx.dtype)
    x = xx * torch.cos(ang) - yy * torch.sin(ang)
    z = torch.ones_like(x) * (0.5 + 0.1 * face_idx)
    y = yy
    u = torch.atan2(x, z) / torch.pi
    v = torch.atan2(y, torch.sqrt(x * x + z * z).clamp_min(1e-6)) / torch.pi
    uv = torch.stack([(u + 1) * 0.5, (v + 1) * 0.5], dim=0)
    return uv.unsqueeze(0)
