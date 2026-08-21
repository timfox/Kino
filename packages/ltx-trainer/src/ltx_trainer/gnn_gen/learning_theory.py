"""Learning-theoretic generalisation bounds (Theorems 1–2; Garg et al., Esser et al.)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def rademacher_complexity_gnn_upper_bound(
    m: int,
    embed_dim: int,
    n_layers: int,
    max_degree: int,
) -> float:
    """Empirical Rademacher complexity scale O(d · L · b / sqrt(m)) [11, Theorem 1]."""
    if m <= 0:
        raise ValueError("m must be positive")
    return embed_dim * n_layers * max_degree / math.sqrt(m)


def generalization_bound_graph_level(
    empirical_risk: float,
    rademacher: float,
    *,
    delta: float = 0.05,
    m: int,
) -> float:
    """R(bf) <= R_emp(bf) + O(Rademacher + sqrt(log(1/delta)/m)) (Theorem 1)."""
    conf = math.sqrt(math.log(1.0 / max(delta, 1e-12)) / m)
    return empirical_risk + rademacher + conf


def transductive_rademacher_scale(m: int, n: int) -> float:
    """Scale factor 1/m + 1/(n-m) in transductive Rademacher (Theorem 2, [16])."""
    if not (0 < m < n):
        raise ValueError("need 0 < m < n for transductive setting")
    return 1.0 / m + 1.0 / (n - m)


def transductive_generalization_bound(
    empirical_risk: float,
    rad_scale: float,
    *,
    m: int,
    n: int,
    delta: float = 0.05,
) -> float:
    """Node-level bound skeleton R(bf) <= R_emp + R_hat + O(max(1/sqrt(m), 1/sqrt(n-m))) (Theorem 2)."""
    tail = max(1.0 / math.sqrt(m), 1.0 / math.sqrt(n - m))
    tail *= math.sqrt(1.0 + math.log(1.0 / max(delta, 1e-12)))
    return empirical_risk + rad_scale + tail


def empirical_rademacher_estimate(
    predictions: Tensor,
    *,
    n_draws: int = 32,
    seed: int = 0,
) -> float:
    """Monte Carlo estimate of sup_f (1/m) sum eps_i f(G_i) for scalar predictions per graph."""
    if predictions.dim() != 1:
        raise ValueError("predictions must be (m,)")
    m = predictions.shape[0]
    gen = torch.Generator(device=predictions.device)
    gen.manual_seed(seed)
    best = 0.0
    for _ in range(n_draws):
        eps = torch.randint(0, 2, (m,), generator=gen, device=predictions.device, dtype=predictions.dtype)
        eps = eps.mul_(2).sub_(1)
        val = (eps * predictions).mean().abs().item()
        best = max(best, val)
    return best
