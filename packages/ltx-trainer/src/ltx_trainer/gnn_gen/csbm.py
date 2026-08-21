"""Contextual stochastic block model (Section 4; Baranwal et al., Fountoulakis et al.)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sample_csbm(
    n: int,
    p: float,
    q: float,
    mu: Tensor,
    *,
    feature_dim: int | None = None,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample (X, A, y) ~ CSBM(n, p, q, mu).

    Labels y_i ∈ {±1}; features X_i ~ N(y_i mu, I); edges within/between communities.
    """
    if not (0.0 <= q < p <= 1.0):
        raise ValueError("require 0 <= q < p <= 1")
    gen = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=gen).mul_(2).sub_(1).to(dtype=torch.float32)
    d0 = feature_dim if feature_dim is not None else mu.numel()
    if mu.numel() != d0:
        mu = mu.view(-1)[:d0]
    x = torch.randn(n, d0, generator=gen) + y.unsqueeze(1) * mu.unsqueeze(0)
    a = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            prob = p if y[i] == y[j] else q
            if torch.rand((), generator=gen).item() < prob:
                a[i, j] = a[j, i] = 1.0
    return x, a, y


def theorem4_gcn_mu_lower_bound(n: int, p: float, q: float) -> float:
    """Scale ω(log n / (sqrt(n)(p+q))) for ||mu|| in Theorem 4 (GCN recovery)."""
    return math.log(max(n, 3)) / (math.sqrt(n) * (p + q))


def theorem4_gat_mu_lower_bound() -> float:
    """ω(sqrt(log n)) feature norm scale for GAT (Theorem 4)."""
    return math.sqrt(math.log(2.0))


def features_linearly_separable_threshold() -> float:
    """ω(sqrt(log n)) for X-only linear recovery (Theorem 4 baseline)."""
    return theorem4_gat_mu_lower_bound()


def graph_signal_to_noise_ratio(n: int, p: float, q: float) -> float:
    """n(p-q)/sqrt(n(p+q)) as in Shi et al. [20] homophily-modulated double descent."""
    denom = math.sqrt(n * (p + q))
    if denom <= 0:
        return 0.0
    return n * (p - q) / denom
