"""Structural entropy from ViT self-attention (Sec. 4.1, Eq. 1–3)."""

from __future__ import annotations

import numpy as np


def row_normalize_attention(attn: np.ndarray, *, eps: float = 1e-8, drop_cls: bool = True) -> np.ndarray:
    """Eq. (1): optionally remove CLS, then row-normalize spatial attention."""
    a = np.asarray(attn, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("attention must be 2D")
    if drop_cls and a.shape[0] == a.shape[1] and a.shape[0] > 1:
        a = a[1:, 1:]
    rowsum = a.sum(axis=1, keepdims=True) + eps
    return a / rowsum


def patch_structural_entropy(attn_row: np.ndarray, *, eps: float = 1e-8) -> float:
    """Eq. (2): entropy of one patch's attention distribution."""
    p = np.asarray(attn_row, dtype=np.float64).reshape(-1)
    p = np.clip(p, eps, 1.0)
    p = p / (p.sum() + eps)
    return float(-np.sum(p * np.log(p + eps)))


def structural_entropy_map(
    attention: np.ndarray,
    *,
    layers: tuple[int, ...] | None = None,
    eps: float = 1e-8,
    drop_cls: bool = True,
) -> np.ndarray:
    """Eq. (3): average entropy over selected layers → e ∈ R^N."""
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError("attention stack must be (L, N, N)")
    layer_idx = list(range(a.shape[0])) if layers is None else [i for i in layers if i < a.shape[0]]
    entropies: list[np.ndarray] = []
    for li in layer_idx:
        norm = row_normalize_attention(a[li], eps=eps, drop_cls=drop_cls)
        ent = np.array([patch_structural_entropy(norm[i], eps=eps) for i in range(norm.shape[0])])
        entropies.append(ent)
    if not entropies:
        return np.zeros(a.shape[1], dtype=np.float64)
    return np.mean(np.stack(entropies, axis=0), axis=0)


def minmax_normalize_entropy(e: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    lo, hi = float(e.min()), float(e.max())
    return (e - lo) / (hi - lo + eps)
