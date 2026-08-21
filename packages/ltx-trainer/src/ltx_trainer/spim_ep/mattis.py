"""Rank-K Mattis decomposition for SPIM coupling matrices (Eq. 5, 9)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.spim_ep.config import PatternMode, SPIMEPConfig


@dataclass
class MattisParams:
    """Trainable θ = {λ_k, ξ_{k,i}}."""

    lambdas: NDArray[np.floating]  # (K,)
    xi: NDArray[np.floating]  # (K, N)

    @property
    def rank(self) -> int:
        return int(self.lambdas.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.xi.shape[1])


def init_mattis(cfg: SPIMEPConfig, *, seed: int = 0) -> tuple[MattisParams, NDArray[np.floating]]:
    """Initialize ξ and λ with variance σ²_J̃ = 1/Nd (Appendix F, Eq. F2)."""
    rng = np.random.default_rng(seed)
    n = cfg.n_total
    k = cfg.rank
    nd = cfg.n_dynamic
    if cfg.pattern_mode == PatternMode.BINARY:
        xi = rng.choice([-1.0, 1.0], size=(k, n)).astype(np.float64)
        lam_std = math.sqrt(2.0 * k / nd)
    else:
        xi = rng.uniform(-0.9, 0.9, size=(k, n)).astype(np.float64)
        lam_std = math.sqrt(k / (0.03645 * nd))
    lambdas = rng.normal(0.0, lam_std, size=k).astype(np.float64)
    j_dyn = build_augmented_j(lambdas, xi, cfg.n_input)
    return MattisParams(lambdas=lambdas, xi=xi), j_dyn


def build_augmented_j(
    lambdas: NDArray[np.floating],
    xi: NDArray[np.floating],
    n_input: int,
) -> NDArray[np.floating]:
    """Symmetric coupling J for augmented state x = (u, s) via Eq. (9)."""
    j_full = coupling_from_mattis(lambdas, xi)
    n = j_full.shape[0]
    j = np.zeros_like(j_full)
    j[:n_input, n_input:] = j_full[:n_input, n_input:]
    j[n_input:, :n_input] = j_full[n_input:, :n_input]
    j[n_input:, n_input:] = j_full[n_input:, n_input:]
    return j


def coupling_from_mattis(
    lambdas: NDArray[np.floating],
    xi: NDArray[np.floating],
) -> NDArray[np.floating]:
    """J_ij = (1/K) Σ_k λ_k ξ_{k,i} ξ_{k,j}."""
    k = lambdas.shape[0]
    j = np.zeros((xi.shape[1], xi.shape[1]), dtype=np.float64)
    for idx in range(k):
        v = lambdas[idx] * xi[idx]
        j += np.outer(v, xi[idx])
    return j / max(k, 1)


def tilde_coupling(j: NDArray[np.floating]) -> NDArray[np.floating]:
    """Effective ˜J from Eq. (A6) for ±π/4 finite-difference gradients."""
    j = np.atleast_2d(np.asarray(j, dtype=np.float64))
    out = j / math.sqrt(2.0)
    diag = np.diag(j)
    for i in range(j.shape[0]):
        out[i, i] = diag[i]
    return out
