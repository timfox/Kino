"""Differentiable bilinear warp for per-pixel shift fields."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def warp_image(img: Tensor, shift: Tensor) -> tuple[Tensor, Tensor]:
    """Warp ``img`` ``[B,C,H,W]`` by ``shift`` ``[B,2,H,W]`` (dx, dy in pixels).

    Returns warped image and validity mask in ``[0,1]``.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
        shift = shift.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    b, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    grid_x = xx + shift[:, 0]
    grid_y = yy + shift[:, 1]
    gx = 2.0 * grid_x / max(w - 1, 1) - 1.0
    gy = 2.0 * grid_y / max(h - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (
        (grid_x >= 0) & (grid_x <= w - 1) & (grid_y >= 0) & (grid_y <= h - 1)
    ).float().unsqueeze(1)
    if squeeze:
        return warped.squeeze(0), valid.squeeze(0)
    return warped, valid
