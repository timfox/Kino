"""Dirichlet parametrization and simplified evidential classifiers (arXiv:2605.22746)."""

from __future__ import annotations

import numpy as np


def softplus_evidence(z: np.ndarray) -> np.ndarray:
    """Monotone evidence map τ(z) = softplus(z)."""
    z = np.asarray(z, dtype=np.float64)
    return np.log1p(np.exp(z))


def exp_evidence(z: np.ndarray) -> np.ndarray:
    """Evidence map τ(z) = exp(z) for softmax parametrization."""
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    return np.exp(z)


def dirichlet_params(
    evidence: np.ndarray,
    prior: float = 1.0,
) -> np.ndarray:
    """ϕ(e) = e + prior (classical EDL) or e when prior=0."""
    e = np.asarray(evidence, dtype=np.float64)
    return e + prior


def project_dirichlet(alpha: np.ndarray) -> np.ndarray:
    """Π(α) = α / α0 (Eq. 3, 9)."""
    a = np.asarray(alpha, dtype=np.float64)
    a0 = float(np.sum(a))
    if a0 <= 0:
        raise ValueError("α0 must be positive")
    return a / a0


def softmax_from_logits(z: np.ndarray) -> np.ndarray:
    """Theorem 1: τ=exp, ϕ=e ⇒ Π(α) equals softmax(z)."""
    e = exp_evidence(z)
    return project_dirichlet(e)


def simplified_classifier(
    z: np.ndarray,
    tau: str = "exp",
    prior: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (α, p̂) for a simplified evidential classifier."""
    if tau == "exp":
        e = exp_evidence(z)
    elif tau == "softplus":
        e = softplus_evidence(z)
    else:
        raise ValueError(f"unknown evidence map: {tau}")
    alpha = dirichlet_params(e, prior=prior)
    p_hat = project_dirichlet(alpha)
    return alpha, p_hat
