"""Spatial Adapter activation maps (Fig. 3 qualitative stub)."""

from __future__ import annotations

import numpy as np


def spatial_activation_map(
    features: np.ndarray,
    *,
    grid: int = 7,
) -> np.ndarray:
    """
    features: (H*W, d) or (d,) — compute ℓ2 norm per spatial cell, min-max to [0,1].

    Returns (grid, grid) heatmap.
    """
    if features.ndim == 1:
        side = int(np.sqrt(features.size))
        if side * side != features.size:
            side = grid
        feat = features.reshape(side, side, -1) if features.size >= grid * grid else None
        if feat is None:
            out = np.zeros((grid, grid))
            out[grid // 2, grid // 2] = 1.0
            return out
        norms = np.linalg.norm(feat, axis=-1)
    else:
        n = features.shape[0]
        side = int(round(np.sqrt(n)))
        if side * side != n:
            side = grid
            padded = np.zeros((grid * grid, features.shape[1]))
            padded[: min(n, padded.shape[0])] = features[: padded.shape[0]]
            features = padded
        feat = features.reshape(side, side, -1)
        norms = np.linalg.norm(feat, axis=-1)
    lo, hi = float(norms.min()), float(norms.max())
    if hi - lo < 1e-9:
        return np.zeros_like(norms)
    normed = (norms - lo) / (hi - lo)
    if normed.shape != (grid, grid):
        from numpy import pad

        if normed.shape[0] < grid:
            pad_h = grid - normed.shape[0]
            normed = np.pad(normed, ((0, pad_h), (0, 0)), mode="edge")
        if normed.shape[1] < grid:
            pad_w = grid - normed.shape[1]
            normed = np.pad(normed, ((0, 0), (0, pad_w)), mode="edge")
        normed = normed[:grid, :grid]
    return normed.astype(np.float64)


def task_activation_demo(
    latent_grid: np.ndarray,
    task: str,
) -> dict[str, np.ndarray]:
    """Down / norm / up activation proxies biased by task name (smoke only)."""
    rng = np.random.default_rng(hash(task) % 2**31)
    base = spatial_activation_map(latent_grid)
    bias = {"steatosis": (2, 3), "ballooning": (3, 4), "inflammation": (1, 5)}.get(task, (3, 3))
    out = np.zeros_like(base)
    out[bias[0], bias[1]] = 1.0
    out = 0.6 * base + 0.4 * out
    down = spatial_activation_map(rng.standard_normal(49 * 8))
    up = spatial_activation_map(rng.standard_normal(49 * 8))
    return {"down": down, "norm": out, "up": up}
