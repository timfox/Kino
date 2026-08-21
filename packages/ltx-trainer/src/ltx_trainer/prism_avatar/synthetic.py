"""Synthetic monocular portrait sequence for CPU smoke."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.prism_avatar.config import PrismAvatarConfig


def synthetic_turn_sequence(
    cfg: PrismAvatarConfig,
    *,
    frames: int = 48,
    seed: int = 0,
) -> dict[str, NDArray[np.floating]]:
    """Generate yaw sweep, alpha mattes, and FLAME-like semantic weights."""
    rng = np.random.default_rng(seed)
    h, w = 128, 96
    t = np.linspace(0, 1, frames)
    yaws = 25.0 * np.sin(2 * np.pi * t)
    positive = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.42, w * 0.5
    positive[(yy - cy) ** 2 + (xx - cx) ** 2 < (min(h, w) * 0.38) ** 2] = 1.0
    risk = np.zeros_like(positive)
    risk[int(h * 0.72) :, :] = 0.6
    alpha = positive * 0.95 + 0.05
    rgb = np.stack([positive * 0.9, positive * 0.75, positive * 0.65], axis=-1)
    mesh = positive.copy()
    return {
        "yaws": yaws.astype(np.float64),
        "rgb": rgb,
        "alpha": alpha,
        "positive": positive,
        "risk": risk,
        "mesh": mesh,
        "scores": rng.uniform(0.4, 1.0, size=frames),
    }
