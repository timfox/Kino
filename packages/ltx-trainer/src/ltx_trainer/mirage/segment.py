"""Dynamic-object and sky exclusion for cache updates (Sec. 4.4)."""
from __future__ import annotations

import numpy as np


def sky_mask_latent(h: int, w: int, *, sky_fraction: float = 0.2) -> np.ndarray:
    """Upper-band sky mask at latent resolution."""
    mask = np.zeros((h, w), dtype=bool)
    rows = max(1, int(h * sky_fraction))
    mask[:rows, :] = True
    return mask


def dynamic_entity_mask_latent(
    h: int,
    w: int,
    *,
    seed: int = 0,
    n_blobs: int = 2,
    blob_radius: int = 3,
) -> np.ndarray:
    """Toy open-vocabulary + segmenter mask: random dynamic blobs."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(n_blobs):
        cy, cx = rng.integers(blob_radius, h - blob_radius), rng.integers(blob_radius, w - blob_radius)
        yy, xx = np.ogrid[:h, :w]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= blob_radius**2
        mask |= disk
    return mask


def build_update_mask(
    h: int,
    w: int,
    *,
    seed: int = 0,
    exclude_sky: bool = True,
    exclude_dynamic: bool = True,
) -> np.ndarray:
    """Union mask Λ_t complement for Eq. 6 (cells excluded from cache update)."""
    mask = np.zeros((h, w), dtype=bool)
    if exclude_sky:
        mask |= sky_mask_latent(h, w)
    if exclude_dynamic:
        mask |= dynamic_entity_mask_latent(h, w, seed=seed)
    return mask
