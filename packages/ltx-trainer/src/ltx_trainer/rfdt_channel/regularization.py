"""LiDAR-guided RF geometric regularization in Blender (§IV-C)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.rfdt_channel.config import RfdtChannelConfig


def align_mesh_to_lidar(
    mesh_vertices: np.ndarray,
    lidar_points: np.ndarray,
) -> dict[str, Any]:
    """Scale + rigid alignment smoke using centroid and extent ratio."""
    mesh_c = mesh_vertices.mean(axis=0)
    lidar_c = lidar_points.mean(axis=0)
    mesh_extent = np.linalg.norm(mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0))
    lidar_extent = np.linalg.norm(lidar_points.max(axis=0) - lidar_points.min(axis=0))
    scale = float(lidar_extent / (mesh_extent + 1e-8))
    return {
        "translation": (lidar_c - mesh_c * scale).tolist(),
        "scale": scale,
        "aligned": True,
    }


def regularization_summary(cfg: RfdtChannelConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RfdtChannelConfig()
    rng = np.random.default_rng(1261)
    mesh = rng.normal(size=(200, 3)).astype(np.float32) * 2
    lidar = mesh * 1.05 + rng.normal(scale=0.02, size=mesh.shape).astype(np.float32)
    return {
        "operations": list(cfg.blender_regularization_ops),
        "alignment": align_mesh_to_lidar(mesh, lidar),
        "outputs": [
            "wall_thickness_solidify",
            "door_window_openings",
            "manifold_topology_repair",
        ],
        "goal": "RF-computable geometry (not visual texture fidelity)",
    }
