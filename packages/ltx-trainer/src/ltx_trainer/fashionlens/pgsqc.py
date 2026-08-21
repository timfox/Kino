"""Proposal-Guided Spherical Query Calibrator (PGSQC) — Sec. IV-C, Eq. (2)–(6)."""

from __future__ import annotations

import math


def l2_normalize(v: list[float], *, eps: float = 1e-8) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) + eps
    return [x / n for x in v]


def dot(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v, strict=True))


def adaptation_proposal(q0: list[float], *, residual_scale: float = 0.15) -> list[float]:
    r"""q_p = Norm(q_0 + q_0 A B) — toy as normalized residual direction (Eq. 2)."""
    pert = [math.sin((i + 1) * residual_scale) for i in range(len(q0))]
    raw = [q0[i] + dot(q0, pert) * 0.1 * pert[i] for i in range(len(q0))]
    return l2_normalize(raw)


def interpolation_lambda(q0: list[float], *, base: float = 0.35, spread: float = 0.2) -> float:
    """λ(q_0) ∈ (0, 1) — toy from representation energy."""
    e = sum(abs(x) for x in q0) / max(len(q0), 1)
    lam = base + spread * (e - 0.3)
    return max(0.05, min(0.95, lam))


def slerp(q0: list[float], qp: list[float], lam: float) -> list[float]:
    r"""Spherical linear interpolation (Eq. 3)."""
    cos_omega = max(-1.0, min(1.0, dot(q0, qp)))
    omega = math.acos(cos_omega)
    if omega < 1e-5:
        return list(q0)
    sin_omega = math.sin(omega)
    w0 = math.sin((1.0 - lam) * omega) / sin_omega
    w1 = math.sin(lam * omega) / sin_omega
    return l2_normalize([w0 * q0[i] + w1 * qp[i] for i in range(len(q0))])


def linear_interp_euclidean(q0: list[float], qp: list[float], lam: float) -> list[float]:
    """Ablation: Euclidean mix (not used in final model)."""
    return l2_normalize([(1.0 - lam) * q0[i] + lam * qp[i] for i in range(len(q0))])


def orthogonality_loss_frobenius(rows: list[list[float]]) -> float:
    r"""‖A^T A - I‖_F^2 for row-wise A (m × r) — toy surrogate for Eq. (5)."""
    if not rows:
        return 0.0
    m = len(rows)
    r = len(rows[0])
    gram = [[0.0] * r for _ in range(r)]
    for row in rows:
        for i in range(r):
            for j in range(r):
                gram[i][j] += row[i] * row[j]
    inv_m = 1.0 / max(m, 1)
    loss = 0.0
    for i in range(r):
        for j in range(r):
            target = 1.0 if i == j else 0.0
            g_ij = gram[i][j] * inv_m
            diff = g_ij - target
            loss += diff * diff
    return loss


def frobenius_squared(mat: list[list[float]]) -> float:
    r"""‖A‖_F^2 + ‖B‖_F^2 style penalty (Eq. 6)."""
    return sum(sum(x * x for x in row) for row in mat)
