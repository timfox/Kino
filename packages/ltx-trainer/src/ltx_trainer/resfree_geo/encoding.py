"""Coordinate-augmented geometric field encoding (Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def encode_field(x_coords: Tensor, param: Tensor) -> Tensor:
    """U(x) = [x, p(x)] stacked as channels (Eq. 10)."""
    if x_coords.shape[-2:] != param.shape[-2:]:
        raise ValueError("coordinate and parameter grids must match spatially")
    return torch.cat([x_coords, param], dim=-3 if x_coords.dim() == 4 else 0)


def random_smooth_field(
    h: int,
    w: int,
    *,
    num_modes: int = 8,
    device: torch.device | None = None,
) -> Tensor:
    """Eq. 11–12: smooth sinusoidal + sparse noise parameter field."""
    device = device or torch.device("cpu")
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    field = torch.zeros(h, w, device=device)
    for _ in range(num_modes):
        ax = torch.empty(1).uniform_(0.1, 2.0)
        fx = 2 ** torch.empty(1).uniform_(-1, 3)
        fy = 2 ** torch.empty(1).uniform_(-1, 3)
        phx = torch.empty(1).uniform_(0, 2 * torch.pi)
        phy = torch.empty(1).uniform_(0, 2 * torch.pi)
        field = field + ax * torch.sin(2 * torch.pi * fx * xx + phx) * torch.cos(2 * torch.pi * fy * yy + phy)
    field = field + 0.05 * torch.randn(h, w, device=device)
    return field


def multi_resolution_stack(x: Tensor) -> Tensor:
    """Eq. 13–15: φ at 1, 1/2, 1/4 resolutions upsampled and concatenated."""
    _, _, h, w = x.shape
    x_half = F.interpolate(x, size=(max(h // 2, 1), max(w // 2, 1)), mode="bilinear", align_corners=False)
    x_quarter = F.interpolate(x, size=(max(h // 4, 1), max(w // 4, 1)), mode="bilinear", align_corners=False)
    up_half = F.interpolate(x_half, size=(h, w), mode="bilinear", align_corners=False)
    up_quarter = F.interpolate(x_quarter, size=(h, w), mode="bilinear", align_corners=False)
    return torch.cat([x, up_half, up_quarter], dim=1)
