"""Segment-aware rotary encoding stub (Sec. 4.3, Appendix C, arXiv:2606.09828)."""
from __future__ import annotations

from enum import Enum

import numpy as np


class FrameRole(str, Enum):
    """Frame tags for segment-aware rotary positional encoding (Appendix C)."""

    NOISY_TARGET = "noisy_target"
    CLEAN_PRECEDING = "clean_preceding"
    CLEAN_REFERENCE = "clean_reference"


ROLE_PHASE: dict[FrameRole, float] = {
    FrameRole.NOISY_TARGET: 0.0,
    FrameRole.CLEAN_PRECEDING: 1.0 / 3.0,
    FrameRole.CLEAN_REFERENCE: 2.0 / 3.0,
}


def frame_role_phases(roles: np.ndarray | list[FrameRole]) -> np.ndarray:
    """Per-frame rotary phase offsets for noisy / preceding / reference tokens."""
    if isinstance(roles, list):
        arr = np.array([ROLE_PHASE[r] for r in roles], dtype=np.float64)
        return arr
    roles_arr = np.asarray(roles)
    out = np.zeros(roles_arr.shape, dtype=np.float64)
    for role, phase in ROLE_PHASE.items():
        out[roles_arr == role.value] = phase
        out[roles_arr == role] = phase
    return out


def segment_rotary_phases(
    visibility: np.ndarray,
    *,
    static_phase: float = 0.0,
    dynamic_phase: float = 0.25,
) -> np.ndarray:
    """Per-token scalar phase: cached (visible) vs hole (dynamic) regions."""
    vis = np.asarray(visibility, dtype=np.float64)
    if vis.ndim == 3:
        vis = vis[0]
    if vis.ndim != 2:
        raise ValueError(f"expected 2D visibility, got {vis.shape}")
    hole = 1.0 - np.clip(vis, 0.0, 1.0)
    return static_phase + dynamic_phase * hole


def _rotate_half(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return np.concatenate([-x2, x1], axis=-1)


def apply_segment_rotary(
    q: np.ndarray,
    k: np.ndarray,
    phases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply visibility-conditioned rotary offsets to query/key (CPU stub)."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    if q.shape != k.shape:
        raise ValueError(f"q/k shape mismatch: {q.shape} vs {k.shape}")
    if phases.shape != q.shape[:-1]:
        if phases.ndim == q.ndim - 2:
            phases = np.broadcast_to(phases[..., None], q.shape[:-1])
        else:
            raise ValueError(f"phase/token shape mismatch: {phases.shape} vs {q.shape[:-1]}")

    theta = phases * np.pi
    while theta.ndim < q.ndim:
        theta = theta[..., None]
    cos = np.cos(theta)
    sin = np.sin(theta)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
