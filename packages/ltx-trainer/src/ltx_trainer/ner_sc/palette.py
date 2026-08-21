"""Learnable color palette for LL sub-band (Eq. 3–4)."""

from __future__ import annotations

import numpy as np


def softmax_palette_weights(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Temperature-scaled softmax over palette index K (Eq. 3)."""
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-8)
    z = z - z.max(axis=0, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-12, None)


def reconstruct_ll_from_palette(
    logits_khw: np.ndarray,
    palette: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convex combination of palette rows → RGB LL (Eq. 4). logits: (K,H,W), palette: (K,3)."""
    if logits_khw.ndim != 3 or palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("logits must be K×H×W and palette K×3")
    k, h, w = logits_khw.shape
    if palette.shape[0] != k:
        raise ValueError("palette rows must match K")
    weights = softmax_palette_weights(logits_khw.reshape(k, -1), temperature=temperature)
    weights = weights.reshape(k, h, w)
    out = np.zeros((h, w, 3), dtype=np.float64)
    for ki in range(k):
        out += weights[ki, :, :, np.newaxis] * palette[ki]
    return out


def logits_nearest_palette(
    ll_rgb: np.ndarray,
    palette: np.ndarray,
    *,
    sharpness: float = 10.0,
) -> np.ndarray:
    """Build logits that softmax to nearest palette color (training-free assignment)."""
    h, w, _ = ll_rgb.shape
    k = palette.shape[0]
    logits = np.full((k, h, w), -sharpness, dtype=np.float64)
    flat = ll_rgb.reshape(-1, 3)
    dists = ((flat[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2).sum(axis=2)
    best = np.argmin(dists, axis=1)
    for idx, ki in enumerate(best):
        i, j = divmod(idx, w)
        logits[ki, i, j] = sharpness
    return logits


def augment_palette_from_ll(ll_rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Seed palette rows from unique LL colors (screen-content discrete prior)."""
    flat = np.round(ll_rgb.reshape(-1, 3), decimals=2)
    uniq = np.unique(flat, axis=0)
    out = palette.copy()
    n = min(len(uniq), out.shape[0])
    out[:n] = uniq[:n]
    return out


def init_screen_palette(k: int = 64, *, seed: int = 0) -> np.ndarray:
    """Initialize palette with discrete UI-like colors for screen content."""
    rng = np.random.default_rng(seed)
    # Mix of grays, primaries, and syntax-highlight hues.
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.2, 0.25],
            [0.9, 0.9, 0.92],
            [0.0, 0.45, 0.85],
            [0.85, 0.2, 0.2],
            [0.1, 0.65, 0.35],
            [0.95, 0.75, 0.1],
        ],
        dtype=np.float64,
    )
    if k <= len(base):
        return base[:k].copy()
    extra = rng.uniform(0.0, 1.0, size=(k - len(base), 3))
    return np.vstack([base, extra])
