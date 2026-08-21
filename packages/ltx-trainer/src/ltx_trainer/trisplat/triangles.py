"""Oriented triangle primitives (TriSplat Sec. 3.1, Eq. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.trisplat.config import TriSplatConfig

# Canonical equilateral template (paper supplement E), pre-scaled ×4
_CANONICAL = np.array(
    [
        [0.0, 0.577, 0.0],
        [-0.5, -0.289, 0.0],
        [0.5, -0.289, 0.0],
    ],
    dtype=np.float64,
) * 4.0


@dataclass
class TrianglePrimitive:
    vertices: np.ndarray  # (3, 3) world
    opacity: float
    blur: float
    color: np.ndarray  # (3,) RGB in [0,1]


@dataclass
class PerPixelAttributes:
    density: np.ndarray  # (H, W)
    scale: np.ndarray  # (H, W, 3)
    color: np.ndarray  # (H, W, 3)
    blur_raw: np.ndarray  # (H, W)


def default_attributes(h: int, w: int, *, seed: int = 0) -> PerPixelAttributes:
    """Heuristic attributes when no trained primitive head is available."""
    rng = np.random.default_rng(seed)
    density = np.full((h, w), 0.65, dtype=np.float32) + 0.1 * rng.standard_normal((h, w)).astype(np.float32)
    density = np.clip(density, 0.2, 0.95)
    scale = np.full((h, w, 3), 2.0, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scale[..., 0] *= 1.0 + 0.1 * (xx / max(w, 1))
    color = np.full((h, w, 3), 0.5, dtype=np.float32)
    blur = np.full((h, w), 0.0, dtype=np.float32)
    return PerPixelAttributes(density=density, scale=scale, color=color, blur_raw=blur)


def _sigmoid_scale(logits: np.ndarray, z: np.ndarray, K: np.ndarray, cfg: TriSplatConfig) -> np.ndarray:
    s = cfg.scale_min + (cfg.scale_max - cfg.scale_min) * (1.0 / (1.0 + np.exp(-logits)))
    pixel_size = 0.5 * (1.0 / K[0, 0] + 1.0 / K[1, 1])
    return (s * z[..., None] * pixel_size * cfg.template_scale).astype(np.float32)


def build_world_triangles(
    pointmap_cam: np.ndarray,
    attrs: PerPixelAttributes,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    K: np.ndarray,
    R_local: np.ndarray,
    normal_mask: np.ndarray,
    *,
    cfg: TriSplatConfig,
    step: int = 100000,
    rgb: np.ndarray | None = None,
) -> list[TrianglePrimitive]:
    """Instantiate oriented triangles at valid pixels (subsampled in export)."""
    from ltx_trainer.trisplat.sharpening import blur_multiplier, map_density_to_opacity, opacity_temperature_scale

    h, w, _ = pointmap_cam.shape
    centers_cam = pointmap_cam
    z = centers_cam[..., 2]
    scale = _sigmoid_scale(attrs.scale, z, K, cfg)
    op = map_density_to_opacity(attrs.density, step, cfg.sharpening)
    op = opacity_temperature_scale(op, step, cfg.sharpening)
    beta = blur_multiplier(step, cfg.sharpening)
    blur = (1.0 / (1.0 + np.exp(-attrs.blur_raw))) * beta + 1e-4

    if rgb is not None:
        attrs.color = np.clip(rgb.astype(np.float32), 0, 1)

    tris: list[TrianglePrimitive] = []
    stride = max(1, cfg.export.subsample_stride)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if not normal_mask[y, x]:
                continue
            o = float(op[y, x])
            if o < cfg.export.opacity_threshold * 0.5:
                continue
            c_cam = centers_cam[y, x]
            c_world = R_cw @ c_cam + t_cw
            Rn = R_local[y, x]
            Rc = R_cw
            s = scale[y, x]
            verts = []
            for k in range(3):
                v_local = _CANONICAL[k] * s
                v_cam = Rn @ v_local
                v_world = Rc @ v_cam + c_world
                verts.append(v_world)
            tris.append(
                TrianglePrimitive(
                    vertices=np.stack(verts, axis=0).astype(np.float32),
                    opacity=o,
                    blur=float(blur[y, x]),
                    color=attrs.color[y, x].copy(),
                )
            )
    return tris
