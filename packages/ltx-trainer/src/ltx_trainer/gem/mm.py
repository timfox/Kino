"""MM-based GEM inference on vMF mixtures with balance regularizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.gem.balance import (
    empirical_mass,
    mixing_balance_grad,
    mixing_balance_value,
)
from ltx_trainer.gem.config import GemConfig
from ltx_trainer.gem.vmf import (
    log_vmf_unnormalized,
    m_step,
    normalize_rows,
    spherical_kmeans_init,
)


@dataclass
class GemFitResult:
    gamma: np.ndarray
    mu: np.ndarray
    kappa: np.ndarray
  # objective trace
    objective: list[float]
    pi: np.ndarray
    n_iters: int


def _entropy_per_point(gamma: np.ndarray, eps: float) -> np.ndarray:
    g = np.clip(gamma, eps, 1.0)
    return -np.sum(g * np.log(g), axis=1)


def gem_objective(
    x: np.ndarray,
    gamma: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    lam: float,
    eps: float,
) -> float:
    """Eq. (3): ELBO fidelity + mixing-balance on π(Γ)."""
    n, k = gamma.shape
    alpha = 1.0 / k
    log_f = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        log_f[:, j] = np.log(alpha + eps) + log_vmf_unnormalized(x, mu[j], kappa[j])
    fidelity = float(np.sum(gamma * log_f))
    entropy = float(_entropy_per_point(gamma, eps).sum())
    pi = empirical_mass(gamma)
    balance = mixing_balance_value(pi, lam)
    return fidelity + entropy + balance


def _e_step_mirror(
    x: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    pi_t: np.ndarray,
    lam: float,
    n_steps: int,
    eps: float,
) -> np.ndarray:
    """Alternating row-softmax + simplex ascent on empirical mass π."""
    n, k = len(x), len(mu)
    u = 1.0 / k
    alpha = 1.0 / k
    log_f = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        log_f[:, j] = np.log(alpha + eps) + log_vmf_unnormalized(x, mu[j], kappa[j])

    gamma = np.full((n, k), 1.0 / k, dtype=np.float64)
    for _ in range(n_steps):
        pi = empirical_mass(gamma)
        # MM surrogate ascent in π (Prop. 3.2): ∇R(π_t) − λ(π − π_t)
        pi_grad = mixing_balance_grad(pi_t, lam) - lam * (pi - pi_t)
        pi = pi + 0.35 * pi_grad
        pi = np.clip(pi, eps, None)
        pi /= pi.sum()
        # Feed balanced π back into row assignments (strength λ, not 1/N)
        logits = log_f + lam * (u - pi)[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        gamma = np.exp(logits)
        gamma /= gamma.sum(axis=1, keepdims=True) + eps
    return gamma


def fit_gem(
    x: np.ndarray,
    cfg: GemConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> GemFitResult:
    """Run GEM MM iterations on ℓ2-normalized rows of *x*."""
    c = cfg or GemConfig()
    rng = rng or np.random.default_rng(0)
    x = normalize_rows(np.asarray(x, dtype=np.float64), eps=c.eps)
    n, _ = x.shape
    k = min(c.n_clusters, n)
    mu = spherical_kmeans_init(x, k, rng)
    kappa = np.full(k, 10.0, dtype=np.float64)
    gamma = np.full((n, k), 1.0 / k, dtype=np.float64)

    objectives: list[float] = []
    prev_obj = -1e30
    for it in range(c.max_mm_iters):
        pi_t = empirical_mass(gamma)
        gamma = _e_step_mirror(
            x, mu, kappa, pi_t, c.balance_lambda, c.e_step_mirror_steps, c.eps
        )
        mu, kappa = m_step(x, gamma, eps=c.eps)
        obj = gem_objective(x, gamma, mu, kappa, c.balance_lambda, c.eps)
        objectives.append(obj)
        if it > 0 and abs(obj - prev_obj) <= c.stop_tol:
            return GemFitResult(
                gamma=gamma,
                mu=mu,
                kappa=kappa,
                objective=objectives,
                pi=empirical_mass(gamma),
                n_iters=it + 1,
            )
        prev_obj = obj
    return GemFitResult(
        gamma=gamma,
        mu=mu,
        kappa=kappa,
        objective=objectives,
        pi=empirical_mass(gamma),
        n_iters=c.max_mm_iters,
    )
