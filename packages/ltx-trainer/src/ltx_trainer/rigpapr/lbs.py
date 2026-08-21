"""Linear blend skinning on PAPR points (Eq. 1, Sec. 3.3)."""

from __future__ import annotations

from typing import Any

import numpy as np


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle ω ∈ R³ → rotation matrix R ∈ SO(3)."""
    w = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-8:
        return np.eye(3)
    k = w / theta
    K = np.array(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def se3_from_rotation(R: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def lbs_deform(
    canonical: np.ndarray,
    weights: np.ndarray,
    bone_transforms: np.ndarray,
    root_translation: np.ndarray | None = None,
) -> np.ndarray:
    """
    Eq. (1): p_i^t = t^t + Σ_b w_{i,b} T_b(R^t) p_i^0.

    canonical: (N, 3), weights: (N, B) softmax, bone_transforms: (B, 4, 4).
    """
    canon = np.asarray(canonical, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    T = np.asarray(bone_transforms, dtype=np.float64)
    n, b = w.shape
    assert canon.shape == (n, 3)
    assert T.shape == (b, 4, 4)
    hom = np.concatenate([canon, np.ones((n, 1))], axis=1)
    out = np.zeros((n, 3), dtype=np.float64)
    for bone_idx in range(b):
        transformed = (T[bone_idx] @ hom.T).T[:, :3]
        out += w[:, bone_idx : bone_idx + 1] * transformed
    if root_translation is not None:
        out += np.asarray(root_translation, dtype=np.float64).reshape(1, 3)
    return out


def softmax_weights(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def idw_transfer(
    mesh_vertices: np.ndarray,
    mesh_weights: np.ndarray,
    points: np.ndarray,
    *,
    k: int = 6,
) -> np.ndarray:
    """Transfer per-vertex skinning weights to PAPR points (App. C.2)."""
    mv = np.asarray(mesh_vertices, dtype=np.float64)
    mw = np.asarray(mesh_weights, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    out = np.zeros((len(pts), mw.shape[1]), dtype=np.float64)
    for i, p in enumerate(pts):
        d = np.linalg.norm(mv - p, axis=1)
        nn = np.argsort(d)[:k]
        inv = 1.0 / (d[nn] + 1e-6)
        coeff = inv / inv.sum()
        out[i] = coeff @ mw[nn]
    return out


def lbs_demo(*, seed: int = 0, num_points: int = 128, num_bones: int = 8) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    canon = rng.standard_normal((num_points, 3)) * 0.5
    logits = rng.standard_normal((num_points, num_bones))
    weights = softmax_weights(logits)
    R = rodrigues(np.array([0.0, 0.3, 0.0]))
    T_b = np.stack([se3_from_rotation(R if j == 1 else np.eye(3)) for j in range(num_bones)])
    deformed = lbs_deform(canon, weights, T_b, root_translation=np.zeros(3))
    disp = float(np.linalg.norm(deformed - canon, axis=1).mean())
    return {
        "num_points": num_points,
        "num_bones": num_bones,
        "mean_displacement": disp,
        "weight_row_sum": float(weights.sum(axis=1).mean()),
    }
