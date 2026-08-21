"""Synthetic equirectangular frames for pano360 smoke (no FFmpeg)."""

from __future__ import annotations

import numpy as np


def synthesize_equirect_rgb(height: int = 128, width: int = 256, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, height, dtype=np.float64)[:, None]
    xx = np.linspace(0, 1, width, dtype=np.float64)[None, :]
    r = (0.4 + 0.5 * yy + 0.05 * rng.standard_normal((height, width))).clip(0, 1)
    g = (0.35 + 0.45 * xx + 0.05 * rng.standard_normal((height, width))).clip(0, 1)
    b = (0.3 + 0.3 * (1 - yy) + 0.05 * rng.standard_normal((height, width))).clip(0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)
