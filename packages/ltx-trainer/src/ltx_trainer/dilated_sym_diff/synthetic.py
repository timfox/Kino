"""Synthetic die-disk masks — Figs. 1–4."""

from __future__ import annotations

import math

import numpy as np

from ltx_trainer.dilated_sym_diff.morphology import dilate_disk


def _disk_mask(h: int, w: int, cy: int, cx: int, rad: int) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    return ((y - cy) ** 2 + (x - cx) ** 2 <= rad * rad).astype(bool)


def die_face_mask(
    h: int = 128,
    w: int = 128,
    *,
    centers: list[tuple[int, int]] | None = None,
    disk_r: int = 12,
) -> np.ndarray:
    """Binary die face with disk pips."""
    if centers is None:
        centers = [(40, 40), (88, 88)]
    out = np.zeros((h, w), dtype=bool)
    for cy, cx in centers:
        out |= _disk_mask(h, w, cy, cx, disk_r)
    return out


def misalign_shift_dilate(
    mask: np.ndarray,
    *,
    dilate_r: int = 3,
    dx: int = 4,
    dy: int = 2,
) -> np.ndarray:
    """Fig. 1 misalignment: dilate then shift."""
    m = dilate_disk(mask, dilate_r)
    out = np.zeros_like(m)
    h, w = m.shape
    sy = max(0, dy)
    sx = max(0, dx)
    ty0, ty1 = sy, h
    sx0, sx1 = sx, w
    fy0, fy1 = 0, h - dy if dy > 0 else h
    fx0, fx1 = 0, w - dx if dx > 0 else w
    if ty1 > ty0 and ty1 - ty0 == fy1 - fy0:
        out[ty0:ty1, sx0:sx1] = m[fy0:fy1, fx0:fx1]
    return out


def delta_align_bound(dilate_r: int, dx: int, dy: int) -> float:
    """δ_align ≈ dilate_r + sqrt(dx² + dy²) — Fig. 1 caption."""
    return dilate_r + math.hypot(dx, dy)


def fig1_reference_mask(a: np.ndarray, b_mis: np.ndarray, *, dilate_r: int = 3) -> np.ndarray:
    """Misalignment-only case: reference is empty (no physical change after realignment)."""
    _ = (a, b_mis, dilate_r)
    return np.zeros_like(a, dtype=bool)


def fig3_die_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Die-3 vs misaligned die-4; reference mask for IoU curve."""
    a = die_face_mask(centers=[(40, 40), (88, 88)])
    b = misalign_shift_dilate(die_face_mask(centers=[(40, 40), (88, 88)]), dilate_r=3, dx=4, dy=2)
    delta = delta_align_bound(3, 4, 2)
    ref = fig1_reference_mask(a, b)
    return a, b, ref, delta
