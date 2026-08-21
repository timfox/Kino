"""Motion History Image helpers (Davis & Bobick, 1997)."""

from __future__ import annotations

import numpy as np


def motion_mask(
    frame_t: np.ndarray,
    frame_t_prev: np.ndarray,
    *,
    theta: float = 20.0,
) -> np.ndarray:
    """Binary motion mask D(x,y,t) per Eq. (1) in the paper."""
    diff = np.abs(frame_t.astype(np.float32) - frame_t_prev.astype(np.float32))
    return (diff > theta).astype(np.float32)


def update_mhi(
    h_prev: np.ndarray,
    mask: np.ndarray,
    *,
    tau: int = 15,
) -> np.ndarray:
    """Single-step MHI update H_τ per Eq. (2)."""
    h = np.maximum(0.0, h_prev - 1.0)
    h = np.where(mask > 0, float(tau), h)
    return h


def mhi_from_sequence(
    frames: np.ndarray,
    *,
    tau: int = 15,
    theta: float = 20.0,
) -> np.ndarray:
    """Compute MHI stack for T grayscale frames (T, H, W)."""
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError("frames must be (T, H, W) with T >= 2")
    t, h, w = frames.shape
    out = np.zeros((t, h, w), dtype=np.float32)
    state = np.zeros((h, w), dtype=np.float32)
    for i in range(1, t):
        mask = motion_mask(frames[i], frames[i - 1], theta=theta)
        state = update_mhi(state, mask, tau=tau)
        out[i] = state
    return out
