"""Lighting features: diffuse/specular maps and ray encoding (Sec. 7.2)."""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def diffuse_maps(
    normals: np.ndarray,
    light_dirs: np.ndarray,
    light_intensities: np.ndarray,
    visibility: np.ndarray,
    *,
    flat_lit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eq. 17: shadow map rho (flat-lit) and diffuse map chi (HDR-lit).

    normals: (N, 3), light_dirs: (L, 3), visibility: (N, L), intensities: (L, 3)
    """
    ndotl = np.clip((normals @ light_dirs.T) * visibility, 0.0, None)  # (N, L)
    rho = ndotl.sum(axis=1)  # (N,)
    if flat_lit:
        return rho.astype(np.float32), np.zeros_like(rho)
    chi = (ndotl * light_intensities[None, :, 0]).sum(axis=1)
    return rho.astype(np.float32), chi.astype(np.float32)


def blinn_phong_importance(
    n: np.ndarray,
    wi: np.ndarray,
    wo: np.ndarray,
    e: np.ndarray,
    vis: float,
    alpha: float,
) -> float:
    """Eq. 19 importance score for specular ray selection."""
    wh = normalize(wi + wo)
    ndoth = max(0.0, float(np.dot(n, wh)))
    ndoti = max(0.0, float(np.dot(n, wi)))
    return (ndoth**alpha) * vis * float(np.linalg.norm(e)) * ndoti


def sample_specular_rays(
    normal: np.ndarray,
    view_dir: np.ndarray,
    light_dirs: np.ndarray,
    light_intensities: np.ndarray,
    visibility: np.ndarray,
    *,
    num_rays: int = 32,
    alpha: float = 64.0,
) -> np.ndarray:
    """Top-r rays by Blinn-Phong importance (Eq. 18–19)."""
    scores = []
    for i, wi in enumerate(light_dirs):
        scores.append(
            blinn_phong_importance(normal, wi, view_dir, light_intensities[i], float(visibility[i]), alpha)
        )
    scores = np.asarray(scores)
    top = np.argsort(scores)[::-1][:num_rays]
    encodings = []
    for i in top:
        wi = light_dirs[i]
        wh = normalize(wi + view_dir)
        encodings.append(
            ray_encoding(
                normal,
                wi,
                view_dir,
                wh,
                visibility=float(visibility[i]),
                intensity=float(np.linalg.norm(light_intensities[i])),
            )
        )
    return np.stack(encodings, axis=0).astype(np.float32)


def ray_encoding(
    normal: np.ndarray,
    wi: np.ndarray,
    wo: np.ndarray,
    wh: np.ndarray,
    *,
    visibility: float,
    intensity: float,
) -> np.ndarray:
    """6D ray encoding psi_p (Eq. 18)."""
    n = normalize(normal)
    return np.array(
        [
            float(np.dot(n, wi)),
            float(np.dot(n, wo)),
            float(np.dot(n, wh)),
            visibility,
            intensity * max(0.0, float(np.dot(n, wi))),
        ],
        dtype=np.float32,
    )
