"""COLMAP → 3DGS → SuGaR visual mesh extraction (§IV-B)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.rfdt_channel.config import RfdtChannelConfig


def gaussian_primitive_fields(num_gaussians: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Smoke: Gi = {μ_i, Σ_i, o_i, c_i} (Eq. 3–5)."""
    rng = np.random.default_rng(seed)
    return {
        "mu": rng.normal(size=(num_gaussians, 3)).astype(np.float32),
        "scale": rng.uniform(0.01, 0.1, size=(num_gaussians, 3)).astype(np.float32),
        "opacity": rng.uniform(0.3, 1.0, size=num_gaussians).astype(np.float32),
        "color": rng.uniform(0, 1, size=(num_gaussians, 3)).astype(np.float32),
    }


def sugar_density_at_point(
    mu: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    x: np.ndarray,
) -> float:
    """D(x) = Σ o_i G_i(x) (Eq. 10)."""
    d = 0.0
    for i in range(len(opacities)):
        diff = x - mu[i]
        inv_var = 1.0 / (scales[i] ** 2 + 1e-6)
        g = np.exp(-0.5 * np.sum(diff**2 * inv_var))
        d += float(opacities[i] * g)
    return d


def reconstruction_pipeline_smoke(cfg: RfdtChannelConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RfdtChannelConfig()
    g = gaussian_primitive_fields(128)
    sample_x = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "pipeline": list(cfg.reconstruction_pipeline),
        "colmap_output": ["poses", "sparse_point_cloud", "undistorted_images"],
        "num_gaussians": len(g["opacity"]),
        "density_sample": sugar_density_at_point(
            g["mu"], g["scale"], g["opacity"], sample_x
        ),
        "mesh_status": "SuGaR_triangular_mesh_extracted",
        "note": "Visual intermediate; requires LiDAR RF regularization before Sionna RT",
    }
