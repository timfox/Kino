"""Egocentric perception: depth guidance and animation (Sec. 6)."""

from __future__ import annotations

import numpy as np


def encode_depth_guidance(
    vertex: np.ndarray,
    vertex_normal: np.ndarray,
    points: np.ndarray,
    point_normals: np.ndarray,
    *,
    epsilon_d: float = 0.05,
    epsilon_n: float = 0.5,
) -> np.ndarray:
    """
    Surface-to-point distance in canonical space (Eq. 9), simplified without skinning.

    Returns scalar guidance per vertex (0 if no valid correspondence).
    """
    if len(points) == 0:
        return np.zeros(3)
    dists = np.linalg.norm(points - vertex[None, :], axis=1)
    j = int(np.argmin(dists))
    if dists[j] > epsilon_d:
        return np.zeros(3)
    if abs(float(np.dot(vertex_normal, point_normals[j]))) < epsilon_n:
        return np.zeros(3)
    return (points[j] - vertex).astype(np.float32)


def unproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
) -> np.ndarray:
    """Eq. 8: depth map to world points."""
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ones = np.ones_like(xs, dtype=np.float64)
    pix = np.stack([xs.ravel(), ys.ravel(), ones.ravel()], axis=0)
    Kinv = np.linalg.inv(K)
    rays_cam = (Kinv @ pix).T
    pts_cam = rays_cam * depth.ravel()[:, None]
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    pts_world = (R @ pts_cam.T).T + t
    return pts_world.astype(np.float32)


def animation_uv_features(
    position_map: np.ndarray,
    normal_map: np.ndarray,
    xi_map: np.ndarray,
) -> np.ndarray:
    """Stack UV maps for AnimationNet input (Fig. 4)."""
    return np.concatenate([position_map, normal_map, xi_map], axis=-1).astype(np.float32)
