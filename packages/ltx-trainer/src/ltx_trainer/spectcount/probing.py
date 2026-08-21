"""Probing detectability grid stub (Fig. 1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.spectcount.config import SpectCountConfig


def probing_grid(
    *,
    time_bins: int = 15,
    freq_bins: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Simulated detection rate grid (baseline weakness pattern)."""
    rng = np.random.default_rng(seed)
    grid = rng.uniform(0.5, 0.95, (time_bins, freq_bins))
    # Early time + mid-frequency weakness (Fig. 1)
    grid[:3, :] *= 0.4
    grid[:, 3:7] *= 0.55
    return np.clip(grid, 0.0, 1.0)


def apply_spectcount_boost(grid: np.ndarray) -> np.ndarray:
    """Post-SpectCount improved detectability."""
    boosted = grid.copy()
    weak = grid < 0.6
    boosted[weak] = np.minimum(1.0, grid[weak] + 0.35)
    return boosted


def probing_demo(*, seed: int = 0, cfg: SpectCountConfig | None = None) -> dict[str, Any]:
    _ = cfg
    base = probing_grid(seed=seed)
    after = apply_spectcount_boost(base)
    return {
        "baseline_mean_detection": float(np.mean(base)),
        "after_spectcount_mean": float(np.mean(after)),
        "early_time_gain": float(np.mean(after[:3]) - np.mean(base[:3])),
        "grid_shape": list(base.shape),
    }
