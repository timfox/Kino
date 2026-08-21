"""S1 topology: circular roll + Siamese shift-equivariance (Sec. 3.4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def roll_azimuth(x: Tensor, delta: int) -> Tensor:
    """Circular shift along ERP width (azimuth / W)."""
    if delta == 0:
        return x
    return torch.roll(x, shifts=delta, dims=-1)


def circular_pad_conv2d_stub(x: Tensor) -> Tensor:
    """Horizontal circular padding before conv (VAE encoder/decoder stub)."""
    if x.dim() != 4:
        return x
    left = x[..., -1:]
    right = x[..., :1]
    return torch.cat([left, x, right], dim=-1)[..., 1:-1]


def siamese_shift_loss(eps_base: Tensor, eps_shifted: Tensor, delta: int) -> Tensor:
    """
    L_shift = ||Roll_δ(ε_base) − ε_shifted||²  (Eq. Siamese consistency).
    """
    rolled = roll_azimuth(eps_base, delta)
    return F.mse_loss(rolled, eps_shifted)


def shifted_position_grid(
    height: int,
    width: int,
    delta: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """
    P_shifted(x, y) = P((x − δ) mod W, y) — positional encoding shift without moving tokens.
    Returns [H, W, 2] normalized coords in [-1, 1].
    """
    dev = device or torch.device("cpu")
    ys = torch.linspace(-1.0, 1.0, height, device=dev)
    xs = torch.linspace(-1.0, 1.0, width, device=dev)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    if delta != 0:
        grid_x = torch.roll(grid_x, shifts=-delta, dims=1)
    return torch.stack([grid_x, grid_y], dim=-1)
