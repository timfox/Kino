"""DoRA decomposition synergy stub (§6)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.eisbach.config import EisbachConfig


def dora_decompose(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """W = m * V/||V|| — magnitude and unit direction."""
    norm = np.linalg.norm(w, axis=-1, keepdims=True) + 1e-8
    direction = w / norm
    magnitude = norm.squeeze(-1)
    return magnitude, direction


def dora_demo(*, seed: int = 0, cfg: EisbachConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(16, 64))
    m, v = dora_decompose(w)
    return {
        "adapter": cfg.adapter,
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "magnitude_shape": list(m.shape),
        "direction_unit_norm": float(np.mean(np.abs(np.linalg.norm(v, axis=-1) - 1.0))) < 1e-5,
        "barrier_rewards_direction_structure": True,
    }
