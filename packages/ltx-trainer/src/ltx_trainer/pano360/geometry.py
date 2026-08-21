"""Equirectangular ↔ pinhole geometry (pure NumPy)."""

from __future__ import annotations

import math

import numpy as np


def is_equirectangular_size(width: int, height: int, *, aspect_tol: float = 0.08) -> bool:
    """True when width:height ≈ 2:1 (standard 360° equirect)."""
    if height < 32 or width < 64:
        return False
    return abs((width / height) - 2.0) <= aspect_tol


def rotation_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World-from-camera rotation (yaw about +Y, then pitch about +X). Shape ``(3, 3)``."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    return ry @ rx


def equirect_pixel_to_direction(u: float, v: float, width: int, height: int) -> np.ndarray:
    """Unit direction for equirect pixel ``(u, v)`` (longitude-latitude)."""
    lon = (u / max(width - 1, 1)) * 2.0 * math.pi - math.pi
    lat = math.pi * 0.5 - (v / max(height - 1, 1)) * math.pi
    clat = math.cos(lat)
    return np.array([clat * math.sin(lon), math.sin(lat), clat * math.cos(lon)], dtype=np.float64)


def direction_to_equirect_uv(direction: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Map unit direction to continuous equirect ``(u, v)`` pixel coordinates."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    lon = math.atan2(float(d[0]), float(d[2]))
    lat = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    u = (lon / math.pi + 1.0) * 0.5 * (width - 1)
    v = (0.5 - lat / math.pi) * (height - 1)
    return u, v


def pinhole_ray_grid(
    out_w: int,
    out_h: int,
    fov_deg: float,
) -> np.ndarray:
    """Camera-space unit rays for each output pixel. Shape ``(out_h, out_w, 3)``."""
    fov = math.radians(fov_deg)
    fx = 0.5 * out_w / math.tan(fov * 0.5)
    fy = fx
    cx = (out_w - 1) * 0.5
    cy = (out_h - 1) * 0.5
    xs = np.arange(out_w, dtype=np.float64)
    ys = np.arange(out_h, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)
    rays = np.stack([(xg - cx) / fx, (yg - cy) / fy, np.ones_like(xg)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-12
    return rays


def build_equirect_remap(
    equirect_hw: tuple[int, int],
    out_hw: tuple[int, int],
    *,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(map_x, map_y)`` for ``cv2.remap`` from equirect source to pinhole view."""
    eh, ew = equirect_hw
    oh, ow = out_hw
    rot = rotation_yaw_pitch(yaw_deg, pitch_deg)
    cam_rays = pinhole_ray_grid(ow, oh, fov_deg).reshape(-1, 3)
    world = (rot @ cam_rays.T).T
    map_x = np.zeros((oh, ow), dtype=np.float32)
    map_y = np.zeros((oh, ow), dtype=np.float32)
    for i, d in enumerate(world):
        u, v = direction_to_equirect_uv(d, ew, eh)
        row, col = divmod(i, ow)
        map_x[row, col] = u
        map_y[row, col] = v
    return map_x, map_y
