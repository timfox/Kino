"""Taylor approximate log-likelihood — Eq. (3.5–3.7)."""

from __future__ import annotations

import math

import numpy as np

from ltx_trainer.lpm_mcmc.config import LPMMCMCConfig
from ltx_trainer.lpm_mcmc.indexing import multi_indices
from ltx_trainer.lpm_mcmc.link import taylor_coefficient_stub
from ltx_trainer.lpm_mcmc.moments import MomentStore
from ltx_trainer.lpm_mcmc.partition import block_centers


def taylor_error_bound(n: int, *, kappa: int, block_width: float, mg: float, bg: float) -> float:
    """R_{n,κ}(b) — Proposition 6.13."""
    return 0.5 * n * (n - 1) * mg * (bg * block_width) ** (kappa + 1)


def tv_error_bound(
    n: int,
    *,
    kappa: int,
    block_width: float,
    mg: float,
    bg: float,
    epsilon_post: float,
) -> float:
    """Theorem 6.14: e^{2R} - 1 + ε_post (returns inf when bound is trivial)."""
    r = taylor_error_bound(n, kappa=kappa, block_width=block_width, mg=mg, bg=bg)
    x = 2.0 * r
    if x > 700.0:
        return float("inf")
    return math.expm1(x) + epsilon_post


def approximate_log_likelihood(
    store: MomentStore,
    centers: dict[int, np.ndarray],
    *,
    cfg: LPMMCMCConfig,
) -> float:
    """L̃(τ) stub — Eq. (3.6) using bound on |D^α g| instead of true derivatives."""
    k = store.k_blocks
    kappa = store.kappa
    alphas = multi_indices(kappa) if kappa > 1 else [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ]
    total = 0.0
    for s in range(k):
        for t in range(k):
            ys = centers.get(s, np.zeros(2))
            yt = centers.get(t, np.zeros(2))
            for ell in (1, 0):
                for alpha in alphas:
                    coeff = taylor_coefficient_stub(sum(alpha), mg=cfg.mg, bg=cfg.bg) / math.factorial(
                        min(sum(alpha), 12) or 1
                    )
                    m_edge = store.m_edge.get((s, t, alpha), 0.0)
                    m_comp = store.m_complete.get((s, t, alpha), 0.0)
                    m_miss = m_comp - m_edge
                    moment = m_edge if ell == 1 else m_miss
                    # center offset stub: shrink moment when centers differ
                    offset = 1.0 / (1.0 + np.linalg.norm(ys - yt))
                    total += 0.5 * coeff * moment * offset * (1.0 if ell == 1 else 0.5)
    return float(total)
