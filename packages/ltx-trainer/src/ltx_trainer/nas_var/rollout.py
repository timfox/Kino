"""Toy next-acceleration-scale autoregressive rollout."""

from __future__ import annotations

import numpy as np

from ltx_trainer.nas_var.config import NasVarConfig


def argmax_tokens(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def rollout_next_scales(
    base_logits: dict[int, np.ndarray],
    *,
    scales: tuple[int, ...] | None = None,
) -> dict[int, int]:
    """Greedy token pick per scale from independent logits (toy)."""
    scales = scales or NasVarConfig().acceleration_scales
    return {k: argmax_tokens(base_logits[k]) for k in scales}


def fuse_latent_grids(grids: list[np.ndarray]) -> np.ndarray:
    """AQ-VAE-style average of active-scale token maps (same spatial size)."""
    if not grids:
        raise ValueError("need at least one grid")
    stack = np.stack([g.astype(np.float64) for g in grids], axis=0)
    return stack.mean(axis=0)
