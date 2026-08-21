"""Normal refinement and mono-normal bootstrap (TriSplat Sec. 3.2)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.trisplat.config import MonoNormalBootstrapConfig
from ltx_trainer.trisplat.geometry import geometry_normals, orientation_aware_smooth_normals


def refine_normals_conv(
    n_geo: np.ndarray,
    n_smooth: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.25,
) -> np.ndarray:
    """Lightweight refinement (U-Net surrogate): edge-aware blend toward smoothed field."""
    try:
        import cv2  # noqa: PLC0415

        depth_f = depth.astype(np.float32)
        gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-8)
        w = (1.0 - edge)[..., None]
        delta = strength * w * (n_smooth - n_geo)
        n_ref = n_geo + delta
        n_ref = n_ref / (np.linalg.norm(n_ref, axis=-1, keepdims=True) + 1e-8)
        n_ref[~mask] = 0
        return n_ref.astype(np.float32)
    except ImportError:
        n_ref = n_geo + strength * (n_smooth - n_geo)
        n_ref = n_ref / (np.linalg.norm(n_ref, axis=-1, keepdims=True) + 1e-8)
        return n_ref.astype(np.float32)


def bootstrap_alpha(step: int, cfg: MonoNormalBootstrapConfig) -> float:
    if step <= cfg.takeover_steps:
        return 1.0
    if step >= cfg.blend_end_steps:
        return 0.0
    t = (step - cfg.takeover_steps) / max(cfg.blend_end_steps - cfg.takeover_steps, 1)
    return float(0.5 * (1.0 + np.cos(np.pi * t)))


def blend_teacher_normals(
    n_ref: np.ndarray,
    n_teacher: np.ndarray | None,
    mask: np.ndarray,
    *,
    step: int,
    cfg: MonoNormalBootstrapConfig,
) -> np.ndarray:
    if n_teacher is None:
        return n_ref
    alpha = bootstrap_alpha(step, cfg)
    if alpha <= 0:
        return n_ref
    valid = mask & np.isfinite(n_teacher).all(axis=-1)
    out = n_ref.copy()
    if alpha >= 1.0:
        out[valid] = n_teacher[valid] / (
            np.linalg.norm(n_teacher[valid], axis=-1, keepdims=True) + 1e-8
        )
        return out.astype(np.float32)
    blended = alpha * n_teacher + (1.0 - alpha) * n_ref
    blended = blended / (np.linalg.norm(blended, axis=-1, keepdims=True) + 1e-8)
    out[valid] = blended[valid]
    return out.astype(np.float32)


def compute_view_normals(
    pointmap: np.ndarray,
    rgb: np.ndarray,
    *,
    teacher: np.ndarray | None = None,
    step: int = 100000,
    bootstrap_cfg: MonoNormalBootstrapConfig | None = None,
    detach_pointmap: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Full normal pipeline for one view."""
    bootstrap_cfg = bootstrap_cfg or MonoNormalBootstrapConfig()
    pmap = pointmap
    if detach_pointmap:
        pmap = pointmap.copy()
    n_geo, mask = geometry_normals(pmap)
    n_smooth = orientation_aware_smooth_normals(n_geo, mask)
    depth = pmap[..., 2]
    n_ref = refine_normals_conv(n_geo, n_smooth, rgb, depth, mask)
    if teacher is not None:
        t_mask = mask & np.isfinite(teacher).all(axis=-1)
        n_fwd = blend_teacher_normals(n_ref, teacher, t_mask, step=step, cfg=bootstrap_cfg)
    else:
        n_fwd = n_ref
    return n_fwd, mask
