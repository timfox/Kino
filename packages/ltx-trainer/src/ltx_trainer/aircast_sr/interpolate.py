"""Temporal interpolation from 6-hourly GraphCast to hourly targets."""

from __future__ import annotations

import numpy as np


def trilinear_time_spatial(
    field: np.ndarray,
    *,
    t_out: int,
    scale_h: float = 1.0,
    scale_w: float = 1.0,
) -> np.ndarray:
    """Trilinear resize on (T, H, W) or (C, T, H, W) tensors using separable linear interp."""
    if field.ndim == 3:
        return _resize_cthw(field[np.newaxis, ...], t_out=t_out, scale_h=scale_h, scale_w=scale_w)[0]
    if field.ndim == 4:
        return _resize_cthw(field, t_out=t_out, scale_h=scale_h, scale_w=scale_w)
    raise ValueError(f"expected 3D or 4D field, got shape {field.shape}")


def _resize_cthw(x: np.ndarray, *, t_out: int, scale_h: float, scale_w: float) -> np.ndarray:
    c, t_in, h, w = x.shape
    t_axis = np.linspace(0, t_in - 1, t_out)
    h_out = max(1, int(round(h * scale_h)))
    w_out = max(1, int(round(w * scale_w)))
    out = np.zeros((c, t_out, h_out, w_out), dtype=np.float64)
    for ci in range(c):
        plane = np.zeros((t_out, h, w), dtype=np.float64)
        for ti, t_src in enumerate(t_axis):
            t0 = int(np.floor(t_src))
            t1 = min(t0 + 1, t_in - 1)
            w1 = t_src - t0
            plane[ti] = (1.0 - w1) * x[ci, t0] + w1 * x[ci, t1]
        out[ci] = _resize_hw_stack(plane, h_out, w_out)
    return out


def _resize_hw_stack(stack: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    t, h, w = stack.shape
    y_axis = np.linspace(0, h - 1, h_out)
    x_axis = np.linspace(0, w - 1, w_out)
    tmp = np.zeros((t, h_out, w), dtype=np.float64)
    for ti in range(t):
        for yi, y_src in enumerate(y_axis):
            y0 = int(np.floor(y_src))
            y1 = min(y0 + 1, h - 1)
            wy = y_src - y0
            row = (1.0 - wy) * stack[ti, y0] + wy * stack[ti, y1]
            tmp[ti, yi] = row
    out = np.zeros((t, h_out, w_out), dtype=np.float64)
    for ti in range(t):
        for yi in range(h_out):
            for xi, x_src in enumerate(x_axis):
                x0 = int(np.floor(x_src))
                x1 = min(x0 + 1, w - 1)
                wx = x_src - x0
                out[ti, yi, xi] = (1.0 - wx) * tmp[ti, yi, x0] + wx * tmp[ti, yi, x1]
    return out
