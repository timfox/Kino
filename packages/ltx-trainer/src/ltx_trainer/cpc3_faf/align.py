"""Temporal preparation and frame alignment (WavLM → Canary timeline)."""

from __future__ import annotations

import numpy as np


def masked_average_downsample(x: np.ndarray, stride: int = 4) -> np.ndarray:
    """Fixed masked average pooling along time (toy WavLM preparation)."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    t = x.shape[0]
    out_len = (t + stride - 1) // stride
    out = np.zeros((out_len, x.shape[1]), dtype=x.dtype)
    for i in range(out_len):
        chunk = x[i * stride : (i + 1) * stride]
        out[i] = chunk.mean(axis=0)
    return out


def conv_stride_downsample(x: np.ndarray, stride: int = 4) -> np.ndarray:
    """Learnable-style conv preparation stub: grouped sum + normalize."""
    pooled = masked_average_downsample(x, stride)
    return pooled / (np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-8)


def adaptive_map_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linear index resampling to match Canary sequence length."""
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if x.shape[0] == target_len:
        return x
    src_idx = np.linspace(0, x.shape[0] - 1, target_len)
    out = np.zeros((target_len, x.shape[1]), dtype=x.dtype)
    for i, s in enumerate(src_idx):
        j0 = int(np.floor(s))
        j1 = min(j0 + 1, x.shape[0] - 1)
        w = s - j0
        out[i] = (1.0 - w) * x[j0] + w * x[j1]
    return out


def temporal_shift(x: np.ndarray, delta_steps: int) -> np.ndarray:
    """Shift prepared WavLM sequence; vacated frames zeroed (paper control)."""
    if delta_steps == 0:
        return x.copy()
    out = np.zeros_like(x)
    if delta_steps > 0:
        out[delta_steps:] = x[:-delta_steps]
    else:
        d = -delta_steps
        out[:-d] = x[d:]
    return out
