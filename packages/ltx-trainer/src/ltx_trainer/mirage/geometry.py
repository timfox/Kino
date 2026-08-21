"""Pinhole geometry for latent spatial memory (Appendix A, Eqs. 4–10)."""
from __future__ import annotations

import numpy as np


def latent_intrinsics(
    k_pixel: np.ndarray,
    *,
    rgb_hw: tuple[int, int],
    latent_hw: tuple[int, int],
) -> np.ndarray:
    """Scale pixel intrinsics to latent resolution (Eq. 7)."""
    h, w = latent_hw
    H, W = rgb_hw
    scale = np.diag([w / W, h / H, 1.0]).astype(np.float64)
    return scale @ np.asarray(k_pixel, dtype=np.float64)


def backproject_cell(
    u: int,
    v: int,
    depth: float,
    k_latent: np.ndarray,
    extrinsics_world_to_cam: np.ndarray,
) -> np.ndarray:
    """Back-project one latent cell to world coordinates (Eq. 8)."""
    if depth <= 0 or not np.isfinite(depth):
        return np.full(3, np.nan, dtype=np.float64)
    k_inv = np.linalg.inv(k_latent)
    pix = np.array([u + 0.5, v + 0.5, 1.0], dtype=np.float64)
    cam = depth * (k_inv @ pix)
    cam_h = np.array([cam[0], cam[1], cam[2], 1.0], dtype=np.float64)
    e_inv = np.linalg.inv(extrinsics_world_to_cam)
    world = e_inv @ cam_h
    return world[:3]


def project_point(
    point_world: np.ndarray,
    k_latent: np.ndarray,
    extrinsics_world_to_cam: np.ndarray,
) -> tuple[int, int, float] | None:
    """Project a world point to latent grid indices (Eq. 9)."""
    p = np.asarray(point_world, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        return None
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    q = extrinsics_world_to_cam @ ph
    if q[2] <= 1e-6:
        return None
    uv1 = k_latent @ q[:3]
    uv1 = uv1 / uv1[2]
    u, v = int(np.floor(uv1[0])), int(np.floor(uv1[1]))
    return u, v, float(q[2])


def omega_set(
    points: list[np.ndarray],
    u: int,
    v: int,
    k_latent: np.ndarray,
    extrinsics: np.ndarray,
) -> list[int]:
    """Candidate memory indices Ω_t(u, v) with positive depth (Eq. 10)."""
    out: list[int] = []
    for idx, p in enumerate(points):
        proj = project_point(p, k_latent, extrinsics)
        if proj is None:
            continue
        pu, pv, depth = proj
        if pu == u and pv == v and depth > 0:
            out.append(idx)
    return out


def admissible_lambda(
    depth: np.ndarray,
    rgb_hw: tuple[int, int],
    *,
    latent_hw: tuple[int, int] | None = None,
    dynamic_mask: np.ndarray | None = None,
    sky_mask: np.ndarray | None = None,
    depth_mode: str = "bilinear",
) -> set[tuple[int, int]]:
    """Admissible latent cells Λ_t for cache update (Eq. 6, Appendix A)."""
    h, w = latent_hw or (
        int(np.ceil(rgb_hw[0] / 16)),
        int(np.ceil(rgb_hw[1] / 16)),
    )
    d_lat = downsample_depth(depth, (h, w), mode=depth_mode)
    cells: set[tuple[int, int]] = set()
    for v in range(h):
        for u in range(w):
            if dynamic_mask is not None and bool(dynamic_mask[v, u]):
                continue
            if sky_mask is not None and bool(sky_mask[v, u]):
                continue
            d = float(d_lat[v, u])
            if np.isfinite(d) and d > 0:
                cells.add((u, v))
    return cells


def downsample_depth(
    depth: np.ndarray,
    target_hw: tuple[int, int],
    mode: str = "bilinear",
) -> np.ndarray:
    """Downsample metric depth to latent grid (Table 5)."""
    src = np.asarray(depth, dtype=np.float64)
    h, w = target_hw
    if src.shape == (h, w):
        return src
    ys = np.linspace(0, src.shape[0] - 1, h)
    xs = np.linspace(0, src.shape[1] - 1, w)
    out = np.zeros((h, w), dtype=np.float64)
    for yi, sy in enumerate(ys):
        for xi, sx in enumerate(xs):
            y0, x0 = int(np.floor(sy)), int(np.floor(sx))
            y1, x1 = min(y0 + 1, src.shape[0] - 1), min(x0 + 1, src.shape[1] - 1)
            fy, fx = sy - y0, sx - x0
            if mode == "nearest":
                out[yi, xi] = src[int(round(sy)), int(round(sx))]
            elif mode == "median":
                patch = src[y0 : y1 + 1, x0 : x1 + 1]
                out[yi, xi] = float(np.median(patch))
            elif mode == "area":
                out[yi, xi] = float(src[y0 : y1 + 1, x0 : x1 + 1].mean())
            else:  # bilinear default
                v00, v01 = src[y0, x0], src[y0, x1]
                v10, v11 = src[y1, x0], src[y1, x1]
                out[yi, xi] = (1 - fy) * (1 - fx) * v00 + (1 - fy) * fx * v01 + fy * (1 - fx) * v10 + fy * fx * v11
    return out
