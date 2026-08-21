"""Snavely reprojection factor and bundle-adjustment residuals (§VI-A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class Camera:
    """Pose + radial distortion calibration (Snavely model)."""

    cam_T_world: Array  # (4, 4)
    focal: float
    k1: float
    k2: float


@dataclass
class Observation:
    cam_idx: int
    point_idx: int
    pixel: Array  # (2,)


def pose_apply(cam_T_world: Array, point: Array) -> Array:
    R = cam_T_world[:3, :3]
    t = cam_T_world[:3, 3]
    return R @ point + t


def snavely_reprojection_residual(
    cam: Camera,
    point: Array,
    pixel_meas: Array,
    *,
    eps: float = 1e-9,
) -> Array:
    p_cam = pose_apply(cam.cam_T_world, point)
    d = p_cam[2]
    d_safe = d + eps * np.sign(d if d != 0 else 1.0)
    p = -p_cam[:2] / d_safe
    p_norm2 = float(np.dot(p, p))
    p_norm4 = p_norm2 * p_norm2
    r = 1.0 + cam.k1 * p_norm2 + cam.k2 * p_norm4
    projected = cam.focal * r * p
    return projected - pixel_meas


def stack_bal_residuals(
    cameras: list[Camera],
    points: Array,
    observations: list[Observation],
) -> Array:
    res: list[Array] = []
    for obs in observations:
        cam = cameras[obs.cam_idx]
        pt = points[obs.point_idx]
        res.append(snavely_reprojection_residual(cam, pt, obs.pixel))
    return np.concatenate(res)


def numerical_jacobian(
    cameras: list[Camera],
    points: Array,
    observations: list[Observation],
    *,
    eps: float = 1e-6,
) -> tuple[Array, Array]:
    """Dense Jacobian stub: perturb translation + focal + each point."""
    r0 = stack_bal_residuals(cameras, points, observations)
    m = len(r0)
    n_params = len(cameras) * 4 + points.size  # t(3)+focal per cam, xyz per point
    J = np.zeros((m, n_params), dtype=np.float64)
    col = 0

    for ci in range(len(cameras)):
        for k in range(3):
            cams = [Camera(c.cam_T_world.copy(), c.focal, c.k1, c.k2) for c in cameras]
            cams[ci].cam_T_world[k, 3] += eps
            rp = stack_bal_residuals(cams, points, observations)
            cams[ci].cam_T_world[k, 3] -= 2 * eps
            rm = stack_bal_residuals(cams, points, observations)
            J[:, col] = (rp - rm) / (2 * eps)
            col += 1
        cams = [Camera(c.cam_T_world.copy(), c.focal, c.k1, c.k2) for c in cameras]
        cams[ci].focal += eps
        rp = stack_bal_residuals(cams, points, observations)
        cams[ci].focal -= 2 * eps
        rm = stack_bal_residuals(cams, points, observations)
        J[:, col] = (rp - rm) / (2 * eps)
        col += 1

    for pi in range(points.shape[0]):
        for k in range(3):
            pts = points.copy()
            pts[pi, k] += eps
            rp = stack_bal_residuals(cameras, pts, observations)
            pts[pi, k] -= 2 * eps
            rm = stack_bal_residuals(cameras, pts, observations)
            J[:, col] = (rp - rm) / (2 * eps)
            col += 1
    return J, r0
