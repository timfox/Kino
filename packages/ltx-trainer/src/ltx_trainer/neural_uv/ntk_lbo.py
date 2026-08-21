"""NTK–LBO subspace alignment diagnostic (Eq. 6, Sec. 3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.neural_uv.lbo import build_vertex_features, lbo_eigenfunctions


def orthonormal_basis(A: np.ndarray) -> np.ndarray:
    """QR orthonormalization (reduced mode — m × min(m,n) columns)."""
    Q, _ = np.linalg.qr(np.asarray(A, dtype=np.float64), mode="reduced")
    return Q


def subspace_alignment(A: np.ndarray, B: np.ndarray, *, r: int) -> float:
    """
    S(r) = (1/r) ||A_r^T B_r||_F^2 (Eq. 6).
    A, B: (N, *) column bases (matched row count).
    """
    if B.shape[1] == 0:
        return 0.0
    r = min(r, A.shape[1], B.shape[1], A.shape[0], B.shape[0])
    if r <= 0:
        return 0.0
    Ar = orthonormal_basis(A[:, :r])
    Br = orthonormal_basis(B[:, :r])
    return float(np.sum((Ar.T @ Br) ** 2) / r)


def ntk_stub(features: np.ndarray, params_flat: np.ndarray, *, subsample: int = 256) -> np.ndarray:
    """
    Toy finite-difference NTK on farthest-point subsample.
    Θ_ij ≈ ⟨∂f/∂θ_i, ∂f/∂θ_j⟩ using random directional derivatives.
    """
    rng = np.random.default_rng(int(abs(hash(params_flat.tobytes())) % (2**31)))
    n = min(subsample, len(features))
    idx = rng.choice(len(features), size=n, replace=False)
    d = len(params_flat)
    k = min(32, d)
    G = rng.standard_normal((k, d))
    j = (G @ params_flat.reshape(-1, 1)).reshape(-1)
    J = np.tile(j.reshape(1, -1), (n, 1))
    return J @ J.T


def ntk_lbo_alignment(
    vertices: np.ndarray,
    faces: np.ndarray,
    params_flat: np.ndarray,
    *,
    k: int = 16,
    ranks: tuple[int, ...] = (4, 8, 16),
    subsample: int = 256,
) -> dict[str, float]:
    feats = build_vertex_features(vertices, faces, k=k)
    n_verts = len(vertices)
    n = min(subsample, n_verts)
    idx = np.linspace(0, n_verts - 1, n, dtype=int)
    if k <= 0:
        return {f"S({r})": 0.0 for r in ranks}
    psi = lbo_eigenfunctions(vertices, faces, k=max(ranks))
    Theta = ntk_stub(feats[idx], params_flat, subsample=n)
    evals, evecs = np.linalg.eigh(Theta)
    order = np.argsort(evals)[::-1]
    Q = evecs[:, order]
    out: dict[str, float] = {}
    for r in ranks:
        out[f"S({r})"] = subspace_alignment(Q, psi[idx], r=r)
    return out


def ntk_lbo_demo(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 48
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.stack([np.cos(angles), np.sin(angles), rng.normal(0, 0.02, n)], axis=1)
    faces = np.array([[0, i, i + 1] for i in range(1, n - 1)], dtype=np.int64)
    params = rng.standard_normal(128)
    k0 = ntk_lbo_alignment(verts, faces, params, k=0, ranks=(4, 8, 16))
    k16 = ntk_lbo_alignment(verts, faces, params, k=16, ranks=(4, 8, 16))
    return {"k0": k0, "k16": k16}
