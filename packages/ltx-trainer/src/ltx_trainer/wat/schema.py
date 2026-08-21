"""Shared wavelet token flatten / scatter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

ModalityName = Literal["audio", "image", "video"]

MODALITY_RANK = {"audio": 1, "image": 2, "video": 3}
MODALITY_ID = {"audio": 0, "image": 1, "video": 2}


@dataclass
class TokenBatch:
    values: Tensor  # (B, N, C)
    modality: Tensor  # (B, N) int
    rank: Tensor
    scale: Tensor
    subband: Tensor
    position: Tensor  # (B, N, 3) normalized (t,y,x)


def coeffs_to_tokens(coeffs: Tensor, modality: ModalityName) -> TokenBatch:
    """Flatten DWT coefficient tensor to dense token schema."""
    rank = MODALITY_RANK[modality]
    b, c_tot, *grid = coeffs.shape
    n_sub = 2**rank
    c = c_tot // n_sub
    g_flat = 1
    for g in grid:
        g_flat *= int(g)
    n = n_sub * g_flat
    t = coeffs.reshape(b, n_sub, c, *grid)
    # (B, n_sub, *grid, C) -> (B, N, C)
    values = t.permute(0, 1, *range(3, 3 + len(grid)), 2).reshape(b, n, c)
    mid = MODALITY_ID[modality]
    device = coeffs.device
    modality_t = torch.full((b, n), mid, dtype=torch.long, device=device)
    rank_t = torch.full((b, n), rank, dtype=torch.long, device=device)
    scale_t = torch.zeros((b, n), dtype=torch.long, device=device)
    subband_ids = torch.arange(n_sub, device=device).repeat_interleave(g_flat)
    subband_t = subband_ids.unsqueeze(0).expand(b, -1)
    base_pos = _base_grid_positions(grid, rank, device=device, dtype=coeffs.dtype)
    position = base_pos.unsqueeze(0).expand(b, -1, -1).clone()
    for sb in range(n_sub):
        sl = slice(sb * g_flat, (sb + 1) * g_flat)
        position[:, sl, :] = base_pos[sl]
    return TokenBatch(values, modality_t, rank_t, scale_t, subband_t, position)


def tokens_to_coeffs(batch: TokenBatch, modality: ModalityName, grid: tuple[int, ...]) -> Tensor:
    """Scatter tokens back to DWT coefficient layout (inverse of coeffs_to_tokens)."""
    rank = MODALITY_RANK[modality]
    n_sub = 2**rank
    b, n, c = batch.values.shape
    if rank == 1:
        g1 = grid[0]
        t = batch.values.reshape(b, n_sub, g1, c).permute(0, 1, 3, 2)
        return t.reshape(b, n_sub * c, g1)
    if rank == 2:
        g1, g2 = grid
        t = batch.values.reshape(b, n_sub, g1, g2, c).permute(0, 1, 4, 2, 3)
        return t.reshape(b, n_sub * c, g1, g2)
    g1, g2, g3 = grid
    t = batch.values.reshape(b, n_sub, g1, g2, g3, c).permute(0, 1, 5, 2, 3, 4)
    return t.reshape(b, n_sub * c, g1, g2, g3)


def _base_grid_positions(
    grid: tuple[int, ...],
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """(N_subband * g_flat, 3) positions before subband tiling."""
    coords: list[tuple[float, float, float]] = []
    if rank == 1:
        (g1,) = grid
        for i in range(g1):
            t = i / max(g1 - 1, 1)
            coords.append((t, 0.0, 0.0))
    elif rank == 2:
        g1, g2 = grid
        for iy in range(g1):
            for ix in range(g2):
                y = iy / max(g1 - 1, 1)
                x = ix / max(g2 - 1, 1)
                coords.append((0.0, y, x))
    else:
        g1, g2, g3 = grid
        for it in range(g1):
            for iy in range(g2):
                for ix in range(g3):
                    t = it / max(g1 - 1, 1)
                    y = iy / max(g2 - 1, 1)
                    x = ix / max(g3 - 1, 1)
                    coords.append((t, y, x))
    return torch.tensor(coords, device=device, dtype=dtype)
