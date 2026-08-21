"""Two-view triangulation for landmark initialization."""

from __future__ import annotations

import numpy as np

from ltx_trainer.event_vo.lie import rot_from_theta


def triangulate_dlt(
    uv0: np.ndarray,
    uv1: np.ndarray,
    p0: np.ndarray,
    theta0: np.ndarray,
    p1: np.ndarray,
    theta1: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    """Linear triangulation from two camera poses and observations."""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)

    def proj_matrix(p: np.ndarray, theta: np.ndarray) -> np.ndarray:
        R = rot_from_theta(theta)
        t = -R @ p
        ext = np.hstack([R, t.reshape(3, 1)])
        return K @ ext

    P0 = proj_matrix(p0, theta0)
    P1 = proj_matrix(p1, theta1)
    x0 = np.array([uv0[0], uv0[1], 1.0])
    x1 = np.array([uv1[0], uv1[1], 1.0])
    A = np.vstack(
        [
            x0[0] * P0[2] - P0[0],
            x0[1] * P0[2] - P0[1],
            x1[0] * P1[2] - P1[0],
            x1[1] * P1[2] - P1[1],
        ]
    )
    _, _, vh = np.linalg.svd(A)
    Xh = vh[-1]
    if abs(Xh[3]) < 1e-9:
        return None
    pt = Xh[:3] / Xh[3]
    if pt[2] < 0.1:
        return None
    return pt.astype(np.float64)
