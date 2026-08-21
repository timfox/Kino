"""Variance, lagged covariance (Prop. 2), and bias–variance (Prop. 3) — Sec. III-C."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def iid_aggregate_variance(per_sample_var: float, K: int) -> float:
    """``Var(\bar{f}) = (1/K) Var(f_t)`` for IID features (Eq. 6)."""
    if K <= 0:
        raise ValueError("K must be positive")
    return per_sample_var / K


def aggregated_covariance_diagonal(gamma0: float, K: int, gammas: list[float] | None = None) -> float:
    """Diagonal trace proxy for Prop. 2: ``tr(Cov(\bar{f})) ≈ (1/K)(Γ_0 + 2 Σ_{τ=1}^{K-1} (1-τ/K) Γ_τ)``."""
    if gammas is None:
        gammas = []
    total = gamma0
    for tau, g in enumerate(gammas, start=1):
        if tau >= K:
            break
        total += 2.0 * (1.0 - tau / K) * g
    return total / K


def empirical_cov_of_aggregate_mean(feats: Tensor) -> Tensor:
    """Finite-sample ``Cov(\bar{f})`` when rows are ``f_t - \bar{f}`` (centered about the row mean).

    For ``feats`` of shape ``[K, D]``, returns ``[D, D]`` matching ``(1/K^2) X^T X`` with row-centered ``X``.
    """
    if feats.dim() != 2:
        raise ValueError(f"feats must be [K, D], got {tuple(feats.shape)}")
    k, _d = feats.shape
    if k <= 0:
        raise ValueError("K must be positive")
    mu = feats.mean(dim=0, keepdim=True)
    x = feats - mu
    return (x.T @ x) / (float(k) ** 2)


def lemma1_output_deviation_squared_bound(L_G: float, cov_bar: Tensor) -> float:
    """Lemma 1: ``E‖G(\bar{f}) - G(μ)‖²`` ≤ ``L_G² tr(Cov(\bar{f}))`` (use trace of ``cov_bar``)."""
    if cov_bar.dim() != 2:
        raise ValueError("cov_bar must be a matrix [D, D]")
    tr = torch.trace(cov_bar).item()
    return float((L_G**2) * tr)


def identity_self_consistency_residual(f: Tensor, feature_mean: Tensor) -> Tensor:
    r"""Per-dimension mean squared ``‖f - \mathbb{E}_t[E(G(f)_t)]‖²`` proxy (Sec. III-B)."""
    return ((f - feature_mean) ** 2).mean()


def bias_variance_prop3_scalar(mu: Tensor, f_bar: Tensor, jacobian_norm: float = 1.0) -> dict[str, float]:
    """Scalar Prop. 3 style decomposition: bias from ``E[\bar{f}] - μ``, variance from ``Var(\bar{f})``."""
    if f_bar.numel() == 0:
        return {"bias": 0.0, "variance": 0.0, "total": 0.0}
    mu_k = f_bar.mean().detach()
    mu_scalar = mu.mean().detach() if mu.numel() else torch.tensor(0.0, dtype=f_bar.dtype, device=f_bar.device)
    bias = float((jacobian_norm * (mu_k - mu_scalar)) ** 2)
    variance = float((jacobian_norm**2) * f_bar.var(unbiased=False).detach())
    return {"bias": bias, "variance": variance, "total": bias + variance}


def bias_variance_decomposition(
    mu: Tensor,
    f_bar: Tensor,
    jacobian_norm: float = 1.0,
) -> dict[str, float]:
    """Alias for :func:`bias_variance_prop3_scalar` (backward compatible)."""
    return bias_variance_prop3_scalar(mu, f_bar, jacobian_norm=jacobian_norm)


def optimal_k_tradeoff(sigma2: float, bias_fn: Callable[[int], float] | None = None) -> int:
    """Discrete search for ``K*`` minimizing ``σ²/K + Bias²(K)`` (Eq. 7; default linear drift proxy)."""
    if bias_fn is None:
        alpha = 0.02

        def bias_fn(k: int) -> float:
            return (alpha * k) ** 2

    best_k, best_obj = 1, float("inf")
    for k in range(1, 11):
        obj = sigma2 / k + bias_fn(k)
        if obj < best_obj:
            best_obj = obj
            best_k = k
    return best_k
