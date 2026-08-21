"""Link functions g1 = log p, g0 = log(1-p) — Eq. (2.3)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ltx_trainer.lpm_mcmc.config import LPMMCMCConfig


def gaussian_edge_prob(
    tau_i: np.ndarray,
    tau_j: np.ndarray,
    *,
    beta0: float,
    beta1: float,
    sigma: float,
) -> float:
    """Gaussian link — Eq. (7.1)."""
    d2 = float(np.sum((tau_i - tau_j) ** 2))
    return beta0 + beta1 * math.exp(-d2 / (2.0 * sigma**2))


def log_link_pair(
    tau_i: np.ndarray,
    tau_j: np.ndarray,
    *,
    edge: bool,
    cfg: LPMMCMCConfig,
) -> float:
    p = gaussian_edge_prob(tau_i, tau_j, beta0=cfg.beta0, beta1=cfg.beta1, sigma=cfg.sigma)
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p) if edge else math.log(1.0 - p)


def exact_log_likelihood(
    tau: np.ndarray,
    edges: set[tuple[int, int]],
    *,
    cfg: LPMMCMCConfig,
) -> float:
    """L(τ) — Eq. (2.3) on directed pairs."""
    n = tau.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            und = (min(i, j), max(i, j))
            total += 0.5 * log_link_pair(tau[i], tau[j], edge=und in edges, cfg=cfg)
    return total


def taylor_coefficient_stub(order: int, *, mg: float, bg: float) -> float:
    """|D^α g| bound stub — Assumption 6.3."""
    return mg * (bg ** order)
