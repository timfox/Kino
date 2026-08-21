"""SSTV, D2CM, Lipschitz ARAB, and pseudo-anomaly (PAG) terms."""

from __future__ import annotations

import numpy as np

from ltx_trainer.ahad.config import AHADParams
from ltx_trainer.ahad.hsi_ops import (
    affine_subspace_basis,
    frobenius_norm,
    grad_c,
    grad_h,
    grad_w,
    mean_shift,
    project_features,
)


def sstv(p: np.ndarray, *, alpha: float, beta: float, gamma: float) -> float:
    """Spectral-spatial total variation (Eq. 5)."""
    return float(
        alpha * np.abs(grad_h(p)).sum()
        + beta * np.abs(grad_w(p)).sum()
        + gamma * np.abs(grad_c(p)).sum()
    )


def d2cm_weights(z: np.ndarray, *, strength: float) -> np.ndarray:
    """D2CM W = η (Z ⊙ Z) (Eq. 8)."""
    peak = float(np.max(np.abs(z))) ** 2
    eta = (peak ** (-1) if peak > 1e-12 else 1.0) * strength
    return eta * (z * z)


def phi_w_lipschitz(z: np.ndarray, w: np.ndarray) -> float:
    """Topology-enhanced ARAB (Eq. 9)."""
    enhanced = z * w
    total = 0.0
    for i in range(enhanced.shape[2]):
        band = enhanced[:, :, i : i + 1]
        total += frobenius_norm(grad_h(band)) + frobenius_norm(grad_w(band))
    return total


def tail_pseudo_anomaly_energy(y: np.ndarray, u: np.ndarray) -> float:
    """Tail energy magnitude (Eq. 10)."""
    y_s, _ = mean_shift(y)
    h, w, c = y_s.shape
    flat = y_s.reshape(-1, c)
    proj = flat @ u @ u.T
    tail = flat - proj
    return frobenius_norm(tail.reshape(h, w, c))


def reg2_arab(y_a: np.ndarray, y_ref: np.ndarray, params: AHADParams) -> float:
    """REG2 = λ1 Φ_W - λ2 tail energy (Eq. 11)."""
    y_s, _ = mean_shift(y_a)
    y_ref_s, _ = mean_shift(y_ref)
    u = affine_subspace_basis(y_ref_s, params.subspace_dim)
    z_a = project_features(y_s, u)
    z_ref = project_features(y_ref_s, u)
    w = d2cm_weights(z_ref, strength=params.d2cm_strength)
    phi = phi_w_lipschitz(z_a, w)
    tail = tail_pseudo_anomaly_energy(y_a, u)
    return params.lambda1 * phi - params.lambda2 * tail
