"""Pinhole projection and measurement Jacobians (Eq. 7–10)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.event_vo.lie import rot_from_theta, skew, transform_point


def project_landmark(
    p_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    X, Y, Z = p_cam
    if Z <= 1e-6:
        Z = 1e-6
    return np.array([fx * X / Z + cx, fy * Y / Z + cy], dtype=np.float64)


def project_with_jacobians(
    p_w: np.ndarray,
    p_c: np.ndarray,
    theta: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (z, H_pos, H_rot, H_lm) each H_* shape (2,3).
    """
    R = rot_from_theta(theta)
    pc = transform_point(R, p_c, p_w)
    z = project_landmark(pc, fx, fy, cx, cy)
    X, Y, Z = pc
    inv_z = 1.0 / max(Z, 1e-6)
    inv_z2 = inv_z * inv_z

    J_pc_p = np.array(
        [
            [fx * inv_z, 0, -fx * X * inv_z2],
            [0, fy * inv_z, -fy * Y * inv_z2],
        ],
        dtype=np.float64,
    )
    H_lm = J_pc_p @ R.T

    pc_cross = skew(pc)
    H_rot = -J_pc_p @ pc_cross

    H_pos = -H_lm
    return z, H_pos, H_rot, H_lm
