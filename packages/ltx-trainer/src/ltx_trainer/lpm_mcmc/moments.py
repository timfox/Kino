"""Block moments M, M^(c), Q — Eq. (3.1–3.2), (4.1–4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ltx_trainer.lpm_mcmc.indexing import multi_indices, projected_indices


def monomial_2d(x: np.ndarray, alpha: tuple[int, int]) -> float:
    return float(x[0] ** alpha[0] * x[1] ** alpha[1])


def monomial_pair(ti: np.ndarray, tj: np.ndarray, alpha: tuple[int, int, int, int]) -> float:
    return monomial_2d(ti, (alpha[0], alpha[1])) * monomial_2d(tj, (alpha[2], alpha[3]))


@dataclass
class MomentStore:
    """Retained sketch for Algorithm 9 (first-order or small κ)."""

    kappa: int
    k_blocks: int
    m_edge: dict[tuple[int, int, tuple[int, int, int, int]], float] = field(default_factory=dict)
    m_complete: dict[tuple[int, int, tuple[int, int, int, int]], float] = field(default_factory=dict)
    q_edge: dict[tuple[int, int, tuple[int, int]], float] = field(default_factory=dict)
    q_complete: dict[tuple[int, tuple[int, int]], float] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        tau: np.ndarray,
        edges: set[tuple[int, int]],
        partition: np.ndarray,
        *,
        kappa: int,
    ) -> MomentStore:
        n = tau.shape[0]
        k = int(partition.max()) + 1
        store = cls(kappa=kappa, k_blocks=k)
        alphas = multi_indices(kappa) if kappa > 1 else [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
        proj = projected_indices(kappa) if kappa > 1 else [(0, 0), (1, 0), (0, 1)]

        for t in range(k):
            idx_t = np.where(partition == t)[0]
            for alpha in proj:
                store.q_complete[(t, alpha)] = sum(monomial_2d(tau[j], alpha) for j in idx_t)

        for i in range(n):
            for t in range(k):
                for alpha in proj:
                    val = 0.0
                    for j in range(n):
                        if i == j:
                            continue
                        und = (min(i, j), max(i, j))
                        if und in edges and partition[j] == t:
                            val += monomial_2d(tau[j], alpha)
                    store.q_edge[(i, t, alpha)] = val

        for s in range(k):
            for t in range(k):
                for alpha in alphas:
                    m_e = 0.0
                    m_c = 0.0
                    for i in range(n):
                        if partition[i] != s:
                            continue
                        for j in range(n):
                            if i == j or partition[j] != t:
                                continue
                            und = (min(i, j), max(i, j))
                            term = monomial_pair(tau[i], tau[j], alpha)
                            m_c += term
                            if und in edges:
                                m_e += term
                    store.m_edge[(s, t, alpha)] = m_e
                    store.m_complete[(s, t, alpha)] = m_c
        return store


def delta_tau_monomial(tau_k: np.ndarray, delta: np.ndarray, alpha: tuple[int, int]) -> float:
    """∆τ_k(δ, α) — Eq. (4.4)."""
    new = tau_k + delta
    return monomial_2d(new, alpha) - monomial_2d(tau_k, alpha)
