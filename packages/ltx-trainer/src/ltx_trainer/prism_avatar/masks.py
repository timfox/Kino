"""Strict head-and-hair supervision domain (Eq. 8–9)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.prism_avatar.config import PrismAvatarConfig


def dilate_binary(mask: NDArray[np.floating], radius: int) -> NDArray[np.floating]:
    """Square dilation stub for head support."""
    if radius <= 0:
        return mask
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            ys = slice(max(0, dy), h + min(0, dy))
            xs = slice(max(0, dx), w + min(0, dx))
            ys_src = slice(max(0, -dy), h - max(0, dy))
            xs_src = slice(max(0, -dx), w - max(0, dx))
            out[ys, xs] = np.maximum(out[ys, xs], mask[ys_src, xs_src])
    return out


def semantic_support_masks(
    positive: NDArray[np.floating],
    risk: NDArray[np.floating],
    cfg: PrismAvatarConfig,
) -> NDArray[np.floating]:
    """Eq. (8): positive minus risk semantic mask."""
    p = (positive > cfg.support_tau_s).astype(np.float64)
    r = (risk > cfg.risk_tau_r).astype(np.float64)
    return p * (1.0 - r)


def head_hair_matte(
    alpha: NDArray[np.floating],
    support: NDArray[np.floating],
    bottom_gate: NDArray[np.floating],
    cfg: PrismAvatarConfig,
) -> NDArray[np.floating]:
    """Eq. (9): strict PMV target matte M_hh."""
    dilated = dilate_binary(support, cfg.hh_dilate_px)
    return alpha * dilated * bottom_gate


def bottom_column_gate(height: int, width: int, cutoff_row: int) -> NDArray[np.floating]:
    """Per-column bottom gate B_t(u) from projected support boundary."""
    gate = np.ones((height, width), dtype=np.float64)
    gate[cutoff_row:, :] = 0.0
    return gate
