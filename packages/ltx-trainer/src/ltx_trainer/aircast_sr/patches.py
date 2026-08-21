"""Patch extraction and cosine-tapered merge (paper § Inference)."""

from __future__ import annotations

import numpy as np


def hann_window_2d(size: int) -> np.ndarray:
    w = np.hanning(size)
    return np.outer(w, w)


def cosine_taper_2d(patch_size: int, overlap: int) -> np.ndarray:
    """2D taper for 50% overlap (stride = patch_size - overlap)."""
    w = hann_window_2d(patch_size)
    return w / np.maximum(w.max(), 1e-8)


def extract_patches(
    field: np.ndarray,
    *,
    patch_size: int,
    stride: int,
) -> list[tuple[tuple[int, int], np.ndarray]]:
    """field (C, H, W) or (C, T, H, W) — returns ((y0, x0), patch)."""
    if field.ndim == 3:
        field = field[:, np.newaxis, ...]
    c, t, h, w = field.shape
    out: list[tuple[tuple[int, int], np.ndarray]] = []
    for y0 in range(0, max(1, h - patch_size + 1), stride):
        for x0 in range(0, max(1, w - patch_size + 1), stride):
            y1 = min(y0 + patch_size, h)
            x1 = min(x0 + patch_size, w)
            patch = np.zeros((c, t, patch_size, patch_size), dtype=field.dtype)
            patch[:, :, : y1 - y0, : x1 - x0] = field[:, :, y0:y1, x0:x1]
            out.append(((y0, x0), patch))
    return out


def merge_patches(
    patches: list[tuple[tuple[int, int], np.ndarray]],
    *,
    out_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """Merge (C, T, H, W) patches into full field with Hann weighting."""
    c = patches[0][1].shape[0]
    t = patches[0][1].shape[1]
    h, w = out_shape
    acc = np.zeros((c, t, h, w), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    taper = cosine_taper_2d(patch_size, patch_size // 2)
    for (y0, x0), patch in patches:
        ph, pw = patch.shape[2], patch.shape[3]
        tw = taper[:ph, :pw]
        acc[:, :, y0 : y0 + ph, x0 : x0 + pw] += patch[:, :, :ph, :pw] * tw
        wsum[y0 : y0 + ph, x0 : x0 + pw] += tw
    wsum = np.maximum(wsum, 1e-8)
    return acc / wsum
