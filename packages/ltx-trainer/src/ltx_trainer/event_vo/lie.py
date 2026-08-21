"""Minimal SO(3)/SE(3) helpers for the ESKF."""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def rot_from_theta(theta: np.ndarray) -> np.ndarray:
    """Rotation matrix from rotation vector (Rodrigues)."""
    th = np.asarray(theta, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(th)
    if angle < 1e-12:
        return np.eye(3)
    k = th / angle
    K = skew(k)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rot_to_theta(R: np.ndarray) -> np.ndarray:
    """Log map SO(3) -> R^3."""
    tr = np.trace(R)
    cos_angle = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-12:
        return np.zeros(3)
    w = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    w /= 2.0 * np.sin(angle)
    return w * angle


def transform_point(R: np.ndarray, t: np.ndarray, p_world: np.ndarray) -> np.ndarray:
    return R.T @ (np.asarray(p_world, dtype=np.float64) - np.asarray(t, dtype=np.float64))
