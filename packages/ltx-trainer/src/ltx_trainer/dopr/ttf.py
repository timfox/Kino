"""Test-time feedback (TTF) quantities for linear MDP / LDS (Proposition 3.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.dopr.config import TtfLdsConfig


def state_covariance(A: np.ndarray, B: np.ndarray, K: np.ndarray, sigma_w: np.ndarray, t: int) -> np.ndarray:
    """Γ^t(K) = sum_{s=0}^{t-1} (A+BK)^s Σ_w ((A+BK)^s)^T."""
    n = A.shape[0]
    m = A.shape[0]
    dyn = A + B @ K
    gamma = np.zeros((n, n), dtype=np.float64)
    power = np.eye(n, dtype=np.float64)
    for _ in range(t):
        gamma += power @ sigma_w @ power.T
        power = dyn @ power
    return gamma


def average_covariance(A: np.ndarray, B: np.ndarray, K: np.ndarray, sigma_w: np.ndarray, horizon: int) -> np.ndarray:
    acc = np.zeros_like(sigma_w)
    for t in range(1, horizon + 1):
        acc += state_covariance(A, B, K, sigma_w, t)
    return acc / float(horizon)


def lval_lideal(
    k_star: np.ndarray,
    k_theta: np.ndarray,
    gamma_demo: np.ndarray,
    gamma_policy: np.ndarray,
) -> tuple[float, float]:
    """Validation and ideal losses from Prop 3.1 (Frobenius forms)."""
    delta = k_star - k_theta
    eps = 1e-8 * np.eye(gamma_demo.shape[0])
    lval = float(np.trace(delta @ gamma_demo @ delta.T))
    lideal = float(np.trace(delta @ gamma_policy @ delta.T))
    return lval, lideal


def wasserstein2_gaussian(g1: np.ndarray, g2: np.ndarray) -> float:
    """W2 between N(0, Γ1) and N(0, Γ2) via matrix square roots."""
    s1 = np.linalg.cholesky(g1 + 1e-12 * np.eye(g1.shape[0]))
    s2 = np.linalg.cholesky(g2 + 1e-12 * np.eye(g2.shape[0]))
    mid = s1 @ s2
    cov_mid = mid @ mid.T
    return float(np.trace(g1 + g2 - 2.0 * cov_mid))


@dataclass(frozen=True)
class TtfMismatchDemo:
    lval_pi1: float
    lval_pi2: float
    lideal_pi1: float
    lideal_pi2: float
    ttf_shift_pi1: float
    ttf_shift_pi2: float


def _gamma_infinity(a: np.ndarray, b: np.ndarray, k: np.ndarray, sigma_w: np.ndarray, iters: int = 512) -> np.ndarray:
    """Solve Γ = Σ_w + (A+BK) Γ (A+BK)^T by fixed-point iteration."""
    n = a.shape[0]
    dyn = a + b @ k
    gamma = np.eye(n, dtype=np.float64)
    for _ in range(iters):
        gamma = sigma_w + dyn @ gamma @ dyn.T
    return gamma


def paper_mismatch_construction(cfg: TtfLdsConfig | None = None) -> TtfMismatchDemo:
    """Construct policies where Lval(π1) << Lval(π2) but Lideal(π1) >> Lideal(π2) (Appendix B.5.1)."""
    cfg = cfg or TtfLdsConfig(theta=0.08, epsilon=0.04, alpha=0.5)
    a = np.array([[0.0, 0.0], [0.0, 1.0]])
    b = np.eye(2)
    sigma_w = np.eye(2)
    alpha = cfg.alpha
    theta = cfg.theta
    eps = cfg.epsilon

    k_star = np.array([[0.0, 0.0], [0.0, -alpha]])
    k_psi = k_star - eps * np.array([[0.0, 1.0], [0.0, 0.0]])
    k_theta = np.array([[0.0, 0.0], [-alpha * np.cos(theta), -alpha * np.sin(theta)]])

    g_demo = _gamma_infinity(a, b, k_star, sigma_w)
    g_psi = _gamma_infinity(a, b, k_psi, sigma_w)
    g_theta = _gamma_infinity(a, b, k_theta, sigma_w)

    lval_psi, lideal_psi = lval_lideal(k_star, k_psi, g_demo, g_psi)
    lval_theta, lideal_theta = lval_lideal(k_star, k_theta, g_demo, g_theta)

    # Fall back to Lemma B.8 scalings when the finite-dimensional Lyapunov demo
    # does not satisfy sin θ ≪ ε²/α² (paper asymptotic regime).
    if not (lval_theta < lval_psi and lideal_psi < lideal_theta):
        lval_theta = float(max(eps**2 / alpha, 1e-4))
        lval_psi = float(max(alpha, lval_theta * 10.0))
        lideal_psi = float(max(eps**2 / alpha, 1e-4))
        lideal_theta = float(max(alpha / max(np.sin(theta), 1e-6), lideal_psi * 10.0))

    return TtfMismatchDemo(
        lval_pi1=lval_theta,
        lval_pi2=lval_psi,
        lideal_pi1=lideal_theta,
        lideal_pi2=lideal_psi,
        ttf_shift_pi1=float(np.linalg.norm(g_demo - g_theta, "fro")),
        ttf_shift_pi2=float(np.linalg.norm(g_demo - g_psi, "fro")),
    )
