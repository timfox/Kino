"""FlatQuant-style affine outlier mitigation stub (arXiv:2606.05861 §III-B)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def init_diagonal_affine(d_out: int, d_in: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Diagonal T with mild scaling to flatten weight dynamic range."""
    rng = rng or np.random.default_rng(0)
    scale = rng.uniform(0.85, 1.15, size=min(d_out, d_in))
    t = np.ones((d_out, d_in), dtype=np.float64)
    n = min(d_out, d_in)
    for i in range(n):
        t[i, i] = scale[i]
    return t


def apply_affine(W: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Left-multiply weights: T^{-1} W (paper notation on linear layers)."""
    w = _as_f64(W)
    t = _as_f64(T)
    if w.ndim == 2 and t.shape[0] == t.shape[1] == w.shape[0]:
        t_inv = np.linalg.inv(t + 1e-6 * np.eye(t.shape[0]))
        return t_inv @ w
    return w / (np.std(w) + 1e-8)


def weight_reconstruction_error(W: np.ndarray, T: np.ndarray, *, quantize_fn) -> float:
    """Relative ||W' - dequant(quant(W'))|| after affine flattening."""
    from ltx_trainer.llmcodec.mapping import rt_dequantize

    w_flat = apply_affine(W, T)
    w_q, scale = quantize_fn(w_flat)
    w_hat = rt_dequantize(w_q, scale)
    return float(np.linalg.norm(w_flat - w_hat) / (np.linalg.norm(w_flat) + 1e-8))


def output_reconstruction_error(
    X: np.ndarray,
    W: np.ndarray,
    T: np.ndarray,
    *,
    quantize_fn,
) -> float:
    """Layer-output Frobenius error proxy for Eq. (1)."""
    return weight_reconstruction_error(W, T, quantize_fn=quantize_fn)


def learn_affine_stub(
    W: np.ndarray,
    *,
    X: np.ndarray | None = None,
    steps: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Search diagonal T minimizing post-quant weight reconstruction error."""
    w = _as_f64(W)
    d_out, d_in = w.shape if w.ndim == 2 else (1, w.size)
    rng = np.random.default_rng(seed)
    from ltx_trainer.llmcodec.mapping import rt_quantize

    def q_fn(arr: np.ndarray):
        return rt_quantize(arr)

    identity = np.eye(d_out, d_in, dtype=np.float64)
    if d_out != d_in:
        identity = np.ones((d_out, d_in), dtype=np.float64)
    best_t = identity
    best_err = weight_reconstruction_error(w, best_t, quantize_fn=q_fn)
    for _ in range(steps):
        cand = init_diagonal_affine(d_out, d_in, rng=rng)
        err = weight_reconstruction_error(w, cand, quantize_fn=q_fn)
        if err < best_err:
            best_t, best_err = cand, err
    return best_t, best_err
