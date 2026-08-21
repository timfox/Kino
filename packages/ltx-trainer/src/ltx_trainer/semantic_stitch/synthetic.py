"""Synthetic overlap pair for CPU smoke."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def synthetic_overlap_pair(
    size: int = 64,
    *,
    seed: int = 0,
) -> dict[str, NDArray[np.floating]]:
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    sun = np.exp(-((xx - w * 0.65) ** 2 + (yy - h * 0.35) ** 2) / (0.08 * min(h, w) ** 2))
    cloud = np.exp(-((xx - w * 0.4) ** 2 + (yy - h * 0.5) ** 2) / (0.12 * min(h, w) ** 2))
    salient = np.clip(sun + 0.6 * cloud, 0, 1)
    i1 = np.stack([0.4 + salient, 0.5 + 0.3 * salient, 0.9 * np.ones((h, w))], axis=-1)
    i2 = i1 * 0.95 + rng.normal(0, 0.02, i1.shape)
    m1 = (sun > 0.3).astype(np.float64)
    m2 = (cloud > 0.25).astype(np.float64)
    grad = np.abs(np.diff(i1.mean(axis=-1), axis=1, prepend=i1.mean(axis=-1)[:, :1]))
    return {
        "i1": np.clip(i1, 0, 1),
        "i2": np.clip(i2, 0, 1),
        "m1": m1,
        "m2": m2,
        "gradient": grad,
    }
