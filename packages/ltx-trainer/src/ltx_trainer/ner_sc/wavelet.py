"""Single-level 2D Haar wavelet (Eq. 1–2, SNeRV backbone)."""

from __future__ import annotations

import numpy as np


def haar_dwt2(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose frame into LL, LH, HL, HH (each H/2 × W/2)."""
    x = np.asarray(frame, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("haar_dwt2 expects H×W grayscale or use per-channel loop")
    h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("height and width must be even")
    x = x.reshape(h // 2, 2, w // 2, 2)
    ll = (x[:, 0, :, 0] + x[:, 0, :, 1] + x[:, 1, :, 0] + x[:, 1, :, 1]) * 0.5
    lh = (x[:, 0, :, 0] + x[:, 0, :, 1] - x[:, 1, :, 0] - x[:, 1, :, 1]) * 0.5
    hl = (x[:, 0, :, 0] - x[:, 0, :, 1] + x[:, 1, :, 0] - x[:, 1, :, 1]) * 0.5
    hh = (x[:, 0, :, 0] - x[:, 0, :, 1] - x[:, 1, :, 0] + x[:, 1, :, 1]) * 0.5
    return ll, lh, hl, hh


def haar_idwt2(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
) -> np.ndarray:
    """Inverse Haar reconstruct full-resolution frame."""
    h2, w2 = ll.shape
    out = np.zeros((h2 * 2, w2 * 2), dtype=np.float64)
    for i in range(h2):
        for j in range(w2):
            a, b, c, d = ll[i, j], lh[i, j], hl[i, j], hh[i, j]
            out[2 * i, 2 * j] = (a + b + c + d) * 0.5
            out[2 * i, 2 * j + 1] = (a + b - c - d) * 0.5
            out[2 * i + 1, 2 * j] = (a - b + c - d) * 0.5
            out[2 * i + 1, 2 * j + 1] = (a - b - c + d) * 0.5
    return out


def haar_dwt2_rgb(frame_hwc: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per-channel Haar; returns 4 arrays shaped (H/2, W/2, 3)."""
    channels = [haar_dwt2(frame_hwc[..., c]) for c in range(frame_hwc.shape[-1])]
    ll = np.stack([t[0] for t in channels], axis=-1)
    lh = np.stack([t[1] for t in channels], axis=-1)
    hl = np.stack([t[2] for t in channels], axis=-1)
    hh = np.stack([t[3] for t in channels], axis=-1)
    return ll, lh, hl, hh


def haar_idwt2_rgb(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
) -> np.ndarray:
    channels = [
        haar_idwt2(ll[..., c], lh[..., c], hl[..., c], hh[..., c]) for c in range(ll.shape[-1])
    ]
    return np.stack(channels, axis=-1)
