"""Depth-normal from ERP depth map (Eq. 11 stub)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.erpgs.erp_projection import distortion_weight_map


def depth_to_normal_erp(depth: Tensor) -> Tensor:
    """
    Finite-difference normals on ERP grid (simplified Eq. 11).
    depth [B,1,H,W] → normal [B,3,H,W] in camera frame.
    """
    d = depth
    dz_du = F.pad(d[..., :, 1:] - d[..., :, :-1], (0, 1, 0, 0))
    dz_dv = F.pad(d[..., 1:, :] - d[..., :-1, :], (0, 0, 0, 1))
    h, w = d.shape[-2:]
    w_map = distortion_weight_map(h, w, device=d.device).view(1, 1, h, w)
    nx = -dz_du * w_map
    ny = -dz_dv * w_map
    nz = torch.ones_like(nx)
    n = torch.cat([nx, ny, nz], dim=1)
    return F.normalize(n, dim=1, eps=1e-6)
