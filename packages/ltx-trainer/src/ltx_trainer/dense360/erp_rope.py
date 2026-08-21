"""ERP-RoPE horizontal index and latitude scaling (Sec. 4.1, Eq. 1–4)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def horizontal_w_index(width: int, device: torch.device | None = None) -> Tensor:
    """
    f(w) satisfying periodicity, boundary consistency, symmetry, max at center (Eq. 3–4).
    Returns [W] float indices in [1, (W+1)/2].
    """
    if width < 1:
        return torch.zeros(0, device=device)
    if width == 1:
        return torch.ones(1, device=device)
    if width % 2 == 1:
        peak = (width + 1) // 2
        up = list(range(1, peak + 1))
        down = list(range(peak - 1, 0, -1))
    else:
        peak = width // 2 + 1
        up = list(range(1, peak + 1))
        down = list(range(width // 2, 1, -1))
    table = up + down
    assert len(table) == width
    return torch.tensor(table, device=device, dtype=torch.float32)


def latitude_gamma(height: int, width: int, device: torch.device | None = None) -> float:
    """γ = H / Σ_θ cos θ (Eq. 2) with θ per ERP row."""
    rows = torch.arange(height, device=device, dtype=torch.float32)
    theta = math.pi / 2.0 - (rows + 0.5) * (math.pi / height)
    cos_sum = torch.cos(theta).sum().item()
    return height / max(cos_sum, 1e-6)


def erp_position_grid(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Per-pixel (g(h), γ·f(w)) for ERP-RoPE.
    Returns row_h [H,W], col_w [H,W].
    """
    gamma = latitude_gamma(height, width, device=device)
    w_idx = horizontal_w_index(width, device=device)
    h_idx = torch.arange(1, height + 1, device=device, dtype=torch.float32)
    col_w = (gamma * w_idx).view(1, width).expand(height, width)
    row_h = h_idx.view(height, 1).expand(height, width)
    return row_h, col_w


def token_erp_coords(
    grid_h: int,
    grid_w: int,
    patch: int,
    full_h: int,
    full_w: int,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Downsampled token centers on ERP position grid."""
    row_h, col_w = erp_position_grid(full_h, full_w, device=device)
    ys = (torch.arange(grid_h, device=device) + 0.5) * patch
    xs = (torch.arange(grid_w, device=device) + 0.5) * patch
    yi = ys.long().clamp(0, full_h - 1)
    xi = xs.long().clamp(0, full_w - 1)
    th = row_h[yi][:, xi]
    tw = col_w[yi][:, xi]
    return th.reshape(-1), tw.reshape(-1)
