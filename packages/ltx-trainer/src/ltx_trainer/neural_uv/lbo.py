"""Laplace–Beltrami spectral features (Eq. 2, Sec. 4)."""

from __future__ import annotations

from typing import Any

import numpy as np


def cotangent_laplacian_stub(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Toy symmetric cotangent Laplacian on a small chart."""
    n = len(vertices)
    L = np.zeros((n, n), dtype=np.float64)
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            w = 1.0 / (np.linalg.norm(vertices[a] - vertices[b]) + 1e-6)
            L[a, b] -= w
            L[b, a] -= w
            L[a, a] += w
            L[b, b] += w
    return L


def lumped_mass(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex lumped mass from one-third adjacent triangle areas."""
    n = len(vertices)
    M = np.zeros(n, dtype=np.float64)
    for tri in faces:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        for idx in tri:
            M[idx] += area / 3.0
    M = np.maximum(M, 1e-8)
    return M


def lbo_eigenfunctions(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    k: int = 16,
) -> np.ndarray:
    """
    First k non-trivial generalized LBO modes (M^{-1/2} L M^{-1/2} eigendecomposition).
    Returns (N, k) with signs fixed by positive correlation to x-coordinate when possible.
    """
    L = cotangent_laplacian_stub(vertices, faces)
    M = lumped_mass(vertices, faces)
    Minv_sqrt = np.diag(1.0 / np.sqrt(M))
    S = Minv_sqrt @ L @ Minv_sqrt
    evals, evecs = np.linalg.eigh(S)
    # skip constant mode
    modes = evecs[:, 1 : k + 1]
    psi = Minv_sqrt @ modes
    # sign fix (App. C.2)
    for j in range(psi.shape[1]):
        corr = float(np.dot(psi[:, j], vertices[:, 0]))
        if corr < 0:
            psi[:, j] *= -1.0
    return psi


def build_vertex_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    k: int = 16,
) -> np.ndarray:
    """xi = [p̃ || ψ_1:k] with per-channel variance normalization."""
    v = np.asarray(vertices, dtype=np.float64)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    p_norm = (v - vmin) / (vmax - vmin + 1e-8)
    psi = lbo_eigenfunctions(v, faces, k=k) if k > 0 else np.zeros((len(v), 0))
    if psi.size:
        psi = psi / (psi.std(axis=0, keepdims=True) + 1e-8)
    return np.concatenate([p_norm, psi], axis=1)


def lbo_demo(*, seed: int = 0, n: int = 32) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    verts = rng.standard_normal((n, 3))
    faces = np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n - 2)], dtype=np.int64)
    feats = build_vertex_features(verts, faces, k=8)
    return {"feature_dim": feats.shape[1], "psi_rank": 8}
