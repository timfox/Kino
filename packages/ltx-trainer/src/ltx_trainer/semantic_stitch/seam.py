"""Soft-coded seam masks and panorama blend (Eq. 1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_pair(l1: NDArray[np.floating], l2: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Ensure L1 + L2 ≈ 1 for soft blending."""
    s = l1 + l2 + 1e-8
    return l1 / s, l2 / s


def stitch_images(
    i1: NDArray[np.floating],
    i2: NDArray[np.floating],
    l1: NDArray[np.floating],
    l2: NDArray[np.floating],
) -> NDArray[np.floating]:
    """S = L1 · I1 + L2 · I2 (Eq. 1)."""
    l1n, l2n = normalize_pair(l1, l2)
    if i1.ndim == 3:
        l1e = l1n[..., None]
        l2e = l2n[..., None]
    else:
        l1e, l2e = l1n, l2n
    return l1e * i1 + l2e * i2


def voronoi_seam(cost: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Baseline Voronoi-like partition from per-pixel cost."""
    pick1 = (cost <= 0.5).astype(np.float64)
    l2 = 1.0 - pick1
    l1 = pick1
    return l1, l2


def graph_cut_seam(gradient: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Stub GC seam: minimum vertical cut on gradient magnitude."""
    h, w = gradient.shape
    dp = gradient.copy()
    for y in range(1, h):
        for x in range(w):
            left = dp[y - 1, max(0, x - 1)]
            mid = dp[y - 1, x]
            right = dp[y - 1, min(w - 1, x + 1)]
            dp[y, x] += min(left, mid, right)
    seam_x = np.zeros(h, dtype=int)
    seam_x[-1] = int(dp[-1].argmin())
    for y in range(h - 2, -1, -1):
        x = seam_x[y + 1]
        seam_x[y] = x + int(np.argmin(dp[y, max(0, x - 1) : min(w, x + 2)]) + max(0, x - 1) - x)
        seam_x[y] = int(np.clip(seam_x[y], 0, w - 1))
    l1 = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        l1[y, : seam_x[y]] = 1.0
    l2 = 1.0 - l1
    return l1, l2
