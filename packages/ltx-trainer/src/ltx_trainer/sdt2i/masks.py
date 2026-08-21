"""ERP mask re-projection stub (Sec. 4.1, Tab. 1)."""

from __future__ import annotations

import torch
from torch import Tensor


def erp_mask_from_perspective(
    center_theta: float,
    center_phi: float,
    fov_deg: float = 120.0,
    erp_h: int = 64,
    erp_w: int = 128,
) -> Tensor:
    """Project perspective FoV center to ERP binary mask (stub)."""
    ys = torch.linspace(-1.5708, 1.5708, erp_h)
    xs = torch.linspace(-3.14159, 3.14159, erp_w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dtheta = (xx - center_theta).abs()
    dphi = (yy - center_phi).abs()
    fov = fov_deg / 180.0 * 3.14159 / 2
    mask = ((dtheta < fov) & (dphi < fov * 0.6)).float()
    return mask.unsqueeze(0).unsqueeze(0)
