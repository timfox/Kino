"""Benign-only cross-model alignment maps (Appendix A)."""

from __future__ import annotations

import numpy as np


def center_columns(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center anchor matrix columns (Eq. 5–6)."""
    mean = h.mean(axis=1, keepdims=True)
    return h - mean, mean.squeeze(axis=1)


def fit_svd_map(hs: np.ndarray, ht: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes W = U V^T (Eq. 14–16)."""
    hs_c, _ = center_columns(hs)
    ht_c, _ = center_columns(ht)
    c = ht_c @ hs_c.T
    u, _, vt = np.linalg.svd(c, full_matrices=False)
    return u @ vt


def fit_ridge_map(hs: np.ndarray, ht: np.ndarray, *, lam: float) -> np.ndarray:
    """Ridge-regularized linear map (Eq. 17–18)."""
    hs_c, _ = center_columns(hs)
    ht_c, _ = center_columns(ht)
    d = hs_c.shape[0]
    return ht_c @ hs_c.T @ np.linalg.inv(hs_c @ hs_c.T + lam * np.eye(d))


def fit_mlp_map(
    hs: np.ndarray,
    ht: np.ndarray,
    *,
    hidden: int = 128,
    steps: int = 200,
    lr: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-hidden-layer GELU mapper (Eq. 19–20 proxy)."""
    rng = rng or np.random.default_rng(0)
    hs_c, _ = center_columns(hs)
    ht_c, _ = center_columns(ht)
    d_in, m = hs_c.shape
    d_out = ht_c.shape[0]
    w1 = rng.normal(scale=0.05, size=(hidden, d_in))
    b1 = np.zeros(hidden)
    w2 = rng.normal(scale=0.05, size=(d_out, hidden))
    b2 = np.zeros(d_out)

    def forward(x: np.ndarray) -> np.ndarray:
        z = w1 @ x + b1[:, np.newaxis]
        h = np.maximum(z, 0.0) + 0.1 * z * (z > 0)  # leaky-ish GELU proxy
        return w2 @ h + b2[:, np.newaxis]

    for _ in range(steps):
        pred = forward(hs_c)
        err = pred - ht_c
        grad_w2 = err @ (w1 @ hs_c + b1[:, np.newaxis]).T / m
        grad_b2 = err.mean(axis=1)
        hidden_act = w1 @ hs_c + b1[:, np.newaxis]
        act = (hidden_act > 0).astype(float)
        back = w2.T @ err
        grad_w1 = back * act @ hs_c.T / m
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
    return w1, b1, w2, b2


def apply_map(
    z: np.ndarray,
    method: str,
    params: object,
) -> np.ndarray:
    if method == "svd":
        w: np.ndarray = params  # type: ignore[assignment]
        return w @ z
    if method == "ridge":
        w = params  # type: ignore[assignment]
        return w @ z
    w1, b1, w2, b2 = params  # type: ignore[misc]
    h = np.maximum(w1 @ z + b1, 0.0)
    return w2 @ h + b2


def median_scale_ratio(hs: np.ndarray, ht: np.ndarray) -> float:
    """β from median anchor norms (Eq. 8)."""
    hs_c, _ = center_columns(hs)
    ht_c, _ = center_columns(ht)
    src = np.median(np.linalg.norm(hs_c, axis=0))
    tgt = np.median(np.linalg.norm(ht_c, axis=0))
    return float(tgt / (src + 1e-8))
