"""Homography-based filter initialization (Sec. III-C, planar scenes)."""

from __future__ import annotations

import numpy as np


def estimate_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """DLT homography mapping src -> dst (N>=4)."""
    n = src.shape[0]
    if n < 4:
        return None
    A = []
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(A, dtype=np.float64))
    H = vh[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-9:
        return None
    return H / H[2, 2]


def homography_to_pose(H: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Recover relative R, t from H = K [r1|r2|t] K^{-1} (planar, normalized plane).
    Returns (R, t) with ||t||=1 (scale ambiguous).
    """
    Kinv = np.linalg.inv(K)
    Hn = Kinv @ H @ K
    h1, h2, h3 = Hn[:, 0], Hn[:, 1], Hn[:, 2]
    lam = 1.0 / np.linalg.norm(h1)
    r1 = h1 * lam
    r2 = h2 * lam
    r3 = np.cross(r1, r2)
    r2 = r2 - r1 * np.dot(r1, r2)
    r2 /= max(np.linalg.norm(r2), 1e-9)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    t = h3 * lam
    return R, t
