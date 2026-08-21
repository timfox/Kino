"""Receiver-centric hemispherical equirectangular projection (Sec. 4.2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def direction_to_equirectangular(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    *,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """
    Map unit direction (x', y', z') to pixel indices (px, py) per paper Sec. 4.2.

    px = (arctan2(z, x) / π + 1) * W / 2
    py = 2 * arcsin(y / ||d||) * H / π
    """
    norm = torch.sqrt(x * x + y * y + z * z).clamp(min=1e-6)
    yn = y / norm
    px = (torch.atan2(z, x) / math.pi + 1.0) * width / 2.0
    py = 2.0 * torch.asin(yn.clamp(-1.0, 1.0)) * height / math.pi
    return px, py


def receiver_view_transform(
    mu: Tensor,
    x_rx: Tensor,
) -> Tensor:
    """Translate Gaussians to receiver-centric coords (identity view stub)."""
    return mu - x_rx.unsqueeze(0)
