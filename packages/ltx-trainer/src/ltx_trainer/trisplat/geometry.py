"""Point maps, geometry normals, and tangent frames (TriSplat Sec. 3.2)."""

from __future__ import annotations

import numpy as np


def pointmap_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """Camera-frame point map ``(H, W, 3)`` from depth ``(H, W)``."""
    h, w = depth.shape
    if u is None or v is None:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def parameterize_pointmap_uvz(pointmap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of Eq. (1): ``p = z * (u, v, 1)``."""
    z = pointmap[..., 2].astype(np.float64)
    z_safe = np.maximum(z, 1e-6)
    u = pointmap[..., 0] / z_safe
    v = pointmap[..., 1] / z_safe
    z_log = np.log(np.maximum(z, 1e-6))
    return u.astype(np.float32), v.astype(np.float32), z_log.astype(np.float32)


def pointmap_from_uvz(u: np.ndarray, v: np.ndarray, z_log: np.ndarray) -> np.ndarray:
    z = np.exp(z_log.astype(np.float64))
    return np.stack([u * z, v * z, z], axis=-1).astype(np.float32)


def geometry_normals(
    pointmap: np.ndarray,
    *,
    toward_camera: bool = True,
    smooth_kernel: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference normals (Eq. 3) and validity mask ``(H, W)``."""
    p = pointmap.astype(np.float64)
    if smooth_kernel > 1:
        try:
            import cv2  # noqa: PLC0415

            k = smooth_kernel | 1
            p = np.stack(
                [cv2.blur(p[..., c], (k, k)) for c in range(3)],
                axis=-1,
            )
        except ImportError:
            pass

    pad = np.pad(p, ((1, 1), (1, 1), (0, 0)), mode="edge")
    dx = pad[1:-1, 2:, :] - pad[1:-1, :-2, :]
    dy = pad[2:, 1:-1, :] - pad[:-2, 1:-1, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = (norm[..., 0] > 1e-8) & np.isfinite(n).all(axis=-1)
    n_out = np.zeros_like(n, dtype=np.float32)
    n_out[valid] = (n[valid] / norm[valid]).astype(np.float32)

    if toward_camera:
        flip = (np.sum(n_out * p, axis=-1) > 0) & valid
        n_out[flip] *= -1.0

    mask = valid.copy()
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return n_out, mask


def orientation_aware_smooth_normals(
    normals: np.ndarray,
    mask: np.ndarray,
    *,
    kernel: int = 5,
) -> np.ndarray:
    """Box filter weighted by normal agreement (paper Sec. 3.2)."""
    h, w, _ = normals.shape
    out = normals.copy()
    r = kernel // 2
    for y in range(r, h - r):
        for x in range(r, w - r):
            if not mask[y, x]:
                continue
            center = normals[y, x]
            acc = np.zeros(3, dtype=np.float64)
            cnt = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ny, nx = y + dy, x + dx
                    if not mask[ny, nx]:
                        continue
                    nb = normals[ny, nx]
                    if np.dot(nb, center) >= 0:
                        acc += nb
                        cnt += 1
            if cnt > 0:
                v = acc / cnt
                vn = np.linalg.norm(v)
                if vn > 1e-8:
                    out[y, x] = (v / vn).astype(np.float32)
    return out


def tangent_frame_from_normal(normal: np.ndarray, delta_x: np.ndarray) -> np.ndarray:
    """Build ``3×3`` rotation ``[t, b, n]`` (columns) for one pixel."""
    n = normal.astype(np.float64)
    n = n / (np.linalg.norm(n) + 1e-8)
    t = delta_x - np.dot(delta_x, n) * n
    tn = np.linalg.norm(t)
    if tn < 1e-8:
        t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t = t - np.dot(t, n) * n
        tn = np.linalg.norm(t) + 1e-8
    t = t / tn
    b = np.cross(n, t)
    t = np.cross(b, n)
    R = np.stack([t, b, n], axis=1)
    return R.astype(np.float32)


def tangent_frames_grid(
    normals: np.ndarray,
    pointmap: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Per-pixel rotation matrices ``(H, W, 3, 3)``."""
    h, w, _ = pointmap.shape
    pad = np.pad(pointmap.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")
    dx = pad[1:-1, 2:, :] - pad[1:-1, :-2, :]
    R_grid = np.zeros((h, w, 3, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                R_grid[y, x] = tangent_frame_from_normal(normals[y, x], dx[y, x])
    return R_grid
