"""Triangle Jacobians, Symmetric Dirichlet, C2 det extension, barrier (Eq. 1, 4–5)."""

from __future__ import annotations

from typing import Any

import numpy as np


def local_frame_coords(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Project triangle vertices to a 2D tangent basis (Eq. 4 local C)."""
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n) + 1e-12
    n = n / n_norm
    u_axis = e1 / (np.linalg.norm(e1) + 1e-12)
    v_axis = np.cross(n, u_axis)
    c0 = np.array([0.0, 0.0])
    c1 = np.array([np.linalg.norm(e1), 0.0])
    c2 = np.array([float(np.dot(e2, u_axis)), float(np.dot(e2, v_axis))])
    return np.stack([c0, c1, c2])


def triangle_jacobian(
    uv: np.ndarray,
    local_c: np.ndarray,
    face: np.ndarray,
) -> np.ndarray:
    """J_t = ΔU_t ΔC_t^{-1} (Eq. 4)."""
    ui, uj, uk = uv[face[0]], uv[face[1]], uv[face[2]]
    dU = np.stack([uj - ui, uk - ui], axis=1)
    dC = np.stack([local_c[1] - local_c[0], local_c[2] - local_c[0]], axis=1)
    return dU @ np.linalg.inv(dC)


def c2_inverse_det_factor(d: float, *, eps_j: float = 1e-4) -> float:
    """Replace d^{-2} with C2 Taylor extension when d ≤ ε_J."""
    if d > eps_j:
        return d ** (-2)
    # matched value, first and second derivative at eps_j
    v0 = eps_j ** (-2)
    v1 = -2.0 * (eps_j ** (-3))
    v2 = 6.0 * (eps_j ** (-4))
    delta = d - eps_j
    return v0 + v1 * delta + 0.5 * v2 * delta * delta


def symmetric_dirichlet_triangle(J: np.ndarray, area: float, *, eps_j: float = 1e-4) -> float:
    """Area-weighted SD term with stable inverse factor."""
    fro = float(np.sum(J * J))
    det = float(np.linalg.det(J))
    q = c2_inverse_det_factor(det, eps_j=eps_j)
    return area * fro * (1.0 + q)


def injectivity_barrier(det: float, *, eps: float = 1e-3) -> float:
    return max(0.0, eps - det) ** 2


def triangle_area_3d(vertices: np.ndarray, face: np.ndarray) -> float:
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))


def chart_objective(
    uv: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    alpha: float = 1.0,
    eps_j: float = 1e-4,
    eps_det: float = 1e-3,
) -> dict[str, float]:
    """Compute E_SD + α Σ [ε - det J]+² and flip rate."""
    esd = 0.0
    barrier = 0.0
    flip_count = 0
    min_det = float("inf")
    for face in faces:
        area = triangle_area_3d(vertices, face)
        lc = local_frame_coords(vertices, face)
        J = triangle_jacobian(uv, lc, face)
        esd += symmetric_dirichlet_triangle(J, area, eps_j=eps_j)
        det = float(np.linalg.det(J))
        min_det = min(min_det, det)
        barrier += injectivity_barrier(det, eps=eps_det)
        if det <= 0:
            flip_count += 1
    n = max(len(faces), 1)
    return {
        "esd": esd,
        "barrier": alpha * barrier,
        "total": esd + alpha * barrier,
        "flip_pct": 100.0 * flip_count / n,
        "min_det": min_det if min_det != float("inf") else 0.0,
    }


def jacobian_demo(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    verts = np.array([[0, 0, 0], [1, 0, 0], [0.3, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv_valid = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    uv_flip = np.array([[0, 0], [1, 0], [0.2, -0.1]], dtype=np.float64)
    return {
        "valid": chart_objective(uv_valid, verts, faces),
        "flipped": chart_objective(uv_flip, verts, faces),
    }
