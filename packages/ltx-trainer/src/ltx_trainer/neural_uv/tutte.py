"""Tutte embedding warm-up (Sec. 3, Algorithm 1 step 3–4)."""

from __future__ import annotations

from typing import Any

import numpy as np


def tutte_embedding(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    boundary: np.ndarray | None = None,
    iters: int = 100,
) -> np.ndarray:
    """
    Discrete Tutte map: interior vertices → convex combination of neighbours;
    boundary pinned to a circle if not supplied.
    """
    n = len(vertices)
    adj: list[set[int]] = [set() for _ in range(n)]
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adj[a].add(b)
            adj[b].add(a)
    if boundary is None:
        # longest boundary loop approximation: vertices with degree != full interior
        deg = np.array([len(adj[i]) for i in range(n)])
        boundary_idx = np.where(deg <= deg.mean())[0]
        if len(boundary_idx) < 3:
            boundary_idx = np.array([0, n // 3, 2 * n // 3])
    else:
        boundary_idx = np.asarray(boundary, dtype=np.int64)
    theta = np.linspace(0, 2 * np.pi, len(boundary_idx), endpoint=False)
    uv = np.zeros((n, 2), dtype=np.float64)
    uv[boundary_idx, 0] = np.cos(theta)
    uv[boundary_idx, 1] = np.sin(theta)
    interior = [i for i in range(n) if i not in set(boundary_idx.tolist())]
    for _ in range(iters):
        new_uv = uv.copy()
        for i in interior:
            nbs = list(adj[i])
            if not nbs:
                continue
            new_uv[i] = uv[nbs].mean(axis=0)
        uv = new_uv
    return uv


def tutte_demo(*, seed: int = 0, n: int = 24) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.stack([np.cos(angles), np.sin(angles), rng.normal(0, 0.05, n)], axis=1)
    faces = np.array([[0, i, i + 1] for i in range(1, n - 1)], dtype=np.int64)
    uv = tutte_embedding(verts, faces)
    from ltx_trainer.neural_uv.jacobian import chart_objective

    metrics = chart_objective(uv, verts, faces, alpha=0.0)
    return {"uv_range": [float(uv.min()), float(uv.max())], **metrics}
