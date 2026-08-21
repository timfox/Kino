"""Mathematical morphology — symmetric and dilated symmetric difference (Eq. 1–2)."""

from __future__ import annotations

from typing import Any

import numpy as np


def complement(mask: np.ndarray) -> np.ndarray:
    return ~mask.astype(bool)


def symmetric_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A△B = (A ∩ B̄) ∪ (B ∩ Ā) — Eq. (1)."""
    aa = a.astype(bool)
    bb = b.astype(bool)
    return (aa & ~bb) | (~aa & bb)


def dilate_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by disk D_r."""
    if radius <= 0:
        return mask.astype(bool).copy()
    m = mask.astype(bool)
    h, w = m.shape
    out = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(m)
    r2 = radius * radius
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        for ny in range(y0, y1):
            for nx in range(x0, x1):
                dy, dx = ny - y, nx - x
                if dy * dy + dx * dx <= r2:
                    out[ny, nx] = True
    return out


def dilated_symmetric_difference(a: np.ndarray, b: np.ndarray, radius: int) -> np.ndarray:
    """A ⊕^r_△ B = (A ∩ (B ⊕ D_r)̄) ∪ (B ∩ (A ⊕ D_r)̄) — Eq. (2)."""
    aa = a.astype(bool)
    bb = b.astype(bool)
    if radius == 0:
        return symmetric_difference(aa, bb)
    bd = dilate_disk(bb, radius)
    ad = dilate_disk(aa, radius)
    return (aa & ~bd) | (bb & ~ad)


def disk_structuring_element(radius: int) -> np.ndarray:
    side = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y <= radius * radius).astype(bool)


def morphology_card() -> dict[str, Any]:
    return {
        "symmetric_difference": "A△B = (A ∩ B̄) ∪ (B ∩ Ā)",
        "dilated_symmetric_difference": "A ⊕^r_△ B with disk D_r",
        "radius_zero": "A ⊕^0_△ B = A△B",
        "structuring_element": "disk D_r",
    }
