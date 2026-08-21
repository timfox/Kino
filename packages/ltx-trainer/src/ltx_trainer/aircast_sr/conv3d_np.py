"""Minimal 3D convolution blocks (NumPy) for AirCast-SR smoke forward passes."""

from __future__ import annotations

import numpy as np


def conv3d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """x: (C_in, D, H, W), weight: (C_out, C_in, kD, kH, kW)."""
    c_out, c_in, kd, kh, kw = weight.shape
    assert x.shape[0] == c_in
    d, h, w = x.shape[1:]
    if padding:
        x = np.pad(x, ((0, 0), (padding,) * 2, (padding,) * 2, (padding,) * 2), mode="edge")
        d, h, w = x.shape[1:]
    od = (d - kd) // stride + 1
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((c_out, od, oh, ow), dtype=np.float64)
    for co in range(c_out):
        for di in range(od):
            for hi in range(oh):
                for wi in range(ow):
                    ds, hs, ws = di * stride, hi * stride, wi * stride
                    patch = x[:, ds : ds + kd, hs : hs + kh, ws : ws + kw]
                    out[co, di, hi, wi] = np.sum(patch * weight[co]) + (0.0 if bias is None else bias[co])
    return out


def group_norm(x: np.ndarray, *, groups: int = 8, eps: float = 1e-5) -> np.ndarray:
    c = x.shape[0]
    g = min(groups, c)
    per = c // g
    out = np.empty_like(x)
    for gi in range(g):
        sl = slice(gi * per, (gi + 1) * per)
        block = x[sl]
        mean = block.mean()
        var = block.var()
        out[sl] = (block - mean) / np.sqrt(var + eps)
    if g * per < c:
        block = x[g * per :]
        mean = block.mean()
        var = block.var()
        out[g * per :] = (block - mean) / np.sqrt(var + eps)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def init_conv_weight(c_out: int, c_in: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(c_in * k**3)
    return rng.standard_normal((c_out, c_in, k, k, k)) * scale
