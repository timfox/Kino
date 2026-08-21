"""Vertex partition from embedding — Appendix A."""

from __future__ import annotations

import numpy as np


def box_partition(
    tau: np.ndarray,
    *,
    block_width: float,
) -> tuple[np.ndarray, int]:
    """Assign nodes to overlapping boxes; return a: [n] -> [K] and K."""
    b = block_width
    step = b / 3.0
    centers_grid = np.arange(-2.0, 2.0 + step, step)
    n = tau.shape[0]
    labels = np.zeros(n, dtype=int)
    used: dict[tuple[float, float], int] = {}
    next_id = 0
    for i in range(n):
        best = None
        best_dist = float("inf")
        for gx in centers_grid:
            for gy in centers_grid:
                cx, cy = gx - b / 2.0, gy - b / 2.0
                dist = max(abs(tau[i, 0] - cx), abs(tau[i, 1] - cy))
                if dist < best_dist:
                    best_dist = dist
                    best = (gx, gy)
        assert best is not None
        if best not in used:
            used[best] = next_id
            next_id += 1
        labels[i] = used[best]
    return labels, next_id


block_partition = box_partition


def block_centers(tau: np.ndarray, partition: np.ndarray) -> dict[int, np.ndarray]:
    """y_s(τ) — Eq. (3.4)."""
    centers: dict[int, np.ndarray] = {}
    for s in np.unique(partition):
        idx = np.where(partition == s)[0]
        centers[int(s)] = tau[idx].mean(axis=0)
    return centers


def is_b_good(tau: np.ndarray, partition: np.ndarray, block_width: float) -> bool:
    """Definition 6.6."""
    for s in np.unique(partition):
        idx = np.where(partition == s)[0]
        if len(idx) < 2:
            continue
        for i in idx:
            for j in idx:
                if np.linalg.norm(tau[i] - tau[j]) > block_width:
                    return False
    return True
