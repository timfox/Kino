"""Continual adaptation (toy) for DMA-KWS stub (arXiv:2605.22120).

Implements LoRA parameter counting and a simple low-rank update composition.
"""

from __future__ import annotations

import numpy as np


def lora_param_count(d: int, r: int, n_mats: int) -> int:
    """Count LoRA params for n matrices of shape (d, d): A(d,r)+B(r,d) each."""
    return int(n_mats * (d * r + r * d))


def apply_lora_update(w: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Return W + scale * (A @ B). Shapes: W(d,d), A(d,r), B(r,d)."""
    w = np.asarray(w, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be square matrix")
    d = w.shape[0]
    if a.shape[0] != d or b.shape[1] != d or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible A/B shapes")
    return w + float(scale) * (a @ b)

