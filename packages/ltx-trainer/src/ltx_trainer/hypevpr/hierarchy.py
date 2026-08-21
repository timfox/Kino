"""Equirectangular hierarchy — progressive FoV halving (Sec. 4.2, Eq. 6, 14)."""

from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor


def num_windows_at_level(level: int) -> int:
    """2^(L-1) windows at finest level L; level 1 has one window."""
    return 2 ** max(level - 1, 0)


def slice_panorama_windows(pano: Tensor, levels: int) -> list[Tensor]:
    """
    Split panorama [B,C,H,Wp] into 2^(L-1) horizontal windows at finest level.
    """
    _, _, _, wp = pano.shape
    n = num_windows_at_level(levels)
    w_each = max(wp // n, 1)
    windows: list[Tensor] = []
    for j in range(n):
        start = j * w_each
        end = wp if j == n - 1 else (j + 1) * w_each
        windows.append(pano[..., start:end])
    return windows


def group_indices(level: int, group_k: int, total_levels: int) -> list[int]:
    """
    I^(ℓ,k)_d from Eq. 14 (1-based j in paper; returns 0-based indices).
    """
    n = num_windows_at_level(total_levels)
    span = 2 ** (total_levels - level)
    start = (group_k - 1) * span
    end = group_k * span
    return list(range(start, min(end, n)))


def iter_level_groups(level: int, total_levels: int) -> Iterator[tuple[int, list[int]]]:
    n_groups = num_windows_at_level(level)
    for k in range(1, n_groups + 1):
        yield k, group_indices(level, k, total_levels)
