"""Image gradients and Laplacian for D3 finite differences (Eq. 11)."""

from __future__ import annotations

import numpy as np


def gradient_xy(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central differences; zero-padded borders."""
    gx = np.zeros_like(img, dtype=np.float64)
    gy = np.zeros_like(img, dtype=np.float64)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    return gx, gy


def laplacian(img: np.ndarray) -> np.ndarray:
    """Discrete Laplacian (isotropic Gaussian pupil proxy)."""
    lap = np.zeros_like(img, dtype=np.float64)
    lap[1:-1, 1:-1] = (
        img[1:-1, 2:]
        + img[1:-1, :-2]
        + img[2:, 1:-1]
        + img[:-2, 1:-1]
        - 4.0 * img[1:-1, 1:-1]
    )
    return lap


def window_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Box-window sum via summed-area table."""
    if radius <= 0:
        return values.astype(np.float64, copy=False)
    r = radius
    k = 2 * r + 1
    padded = np.pad(values.astype(np.float64), r, mode="edge")
    sat = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    sat = np.vstack([np.zeros((1, sat.shape[1])), sat])
    sat = np.hstack([np.zeros((sat.shape[0], 1)), sat])
    h, w = values.shape
    y2 = np.arange(h) + k
    x2 = np.arange(w) + k
    y1 = np.arange(h)
    x1 = np.arange(w)
    return (
        sat[y2[:, None], x2[None, :]]
        - sat[y1[:, None], x2[None, :]]
        - sat[y2[:, None], x1[None, :]]
        + sat[y1[:, None], x1[None, :]]
    )
