"""Toy integrated gradients for 1D temporal attribution (arXiv:2605.23293)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def integrated_gradients_1d(
    x: np.ndarray,
    baseline: np.ndarray,
    grad_fn: Callable[[np.ndarray], float],
    steps: int = 50,
) -> np.ndarray:
    """Riemann-sum IG: (x - x') * mean_t grad F(x' + alpha (x - x'))."""
    x = np.asarray(x, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if x.shape != baseline.shape:
        raise ValueError("x and baseline must have the same shape")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    delta = x - baseline
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:]
    grads = np.zeros_like(x)
    for alpha in alphas:
        path = baseline + alpha * delta
        grads += _numeric_grad(path, grad_fn)
    grads /= float(steps)
    return delta * grads


def _numeric_grad(x: np.ndarray, grad_fn: Callable[[np.ndarray], float], eps: float = 1e-4) -> np.ndarray:
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (grad_fn(xp) - grad_fn(xm)) / (2.0 * eps)
    return g


def aggregate_to_frames(
    attribution: np.ndarray,
    sample_rate_hz: int,
    frame_ms: int = 100,
) -> np.ndarray:
    """Average |attribution| within each frame window."""
    attr = np.abs(np.asarray(attribution, dtype=np.float64))
    frame_samples = max(1, int(sample_rate_hz * frame_ms / 1000))
    n_frames = int(np.ceil(attr.size / frame_samples))
    out = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        start = t * frame_samples
        end = min(attr.size, (t + 1) * frame_samples)
        if start < end:
            out[t] = float(attr[start:end].mean())
    return out


def binarize_percentile(frame_attr: np.ndarray, percentile: float) -> np.ndarray:
    """Binary mask: values >= percentile threshold of frame_attr."""
    frame_attr = np.asarray(frame_attr, dtype=np.float64)
    if frame_attr.size == 0:
        return frame_attr.astype(bool)
    pct = float(np.clip(percentile, 0.0, 100.0))
    thresh = np.percentile(frame_attr, pct)
    return frame_attr >= thresh
