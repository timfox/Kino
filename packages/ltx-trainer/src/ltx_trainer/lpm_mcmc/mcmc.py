"""Metropolis-within-Gibbs embedding update stubs — Algorithms 6–7."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ltx_trainer.lpm_mcmc.config import LPMMCMCConfig
from ltx_trainer.lpm_mcmc.link import exact_log_likelihood
from ltx_trainer.lpm_mcmc.moments import MomentStore, delta_tau_monomial
from ltx_trainer.lpm_mcmc.partition import block_centers
from ltx_trainer.lpm_mcmc.taylor import approximate_log_likelihood


@dataclass
class MCMCState:
    tau: np.ndarray
    partition: np.ndarray
    edges: set[tuple[int, int]]
    store: MomentStore
    approx_loglik: float
    cfg: LPMMCMCConfig


def propose_embedding_step(
    state: MCMCState,
    node: int,
    *,
    step_scale: float = 0.05,
    rng: np.random.Generator,
) -> tuple[bool, MCMCState]:
    """Single MCMC1-style node update using approximate ∆L̃."""
    delta = rng.normal(scale=step_scale, size=2)
    tau_star = state.tau.copy()
    tau_star[node] = tau_star[node] + delta

    centers = block_centers(tau_star, state.partition)
    store_star = MomentStore.build(
        tau_star,
        state.edges,
        state.partition,
        kappa=state.cfg.taylor_order,
    )
    l_tilde_star = approximate_log_likelihood(store_star, centers, cfg=state.cfg)
    delta_l = l_tilde_star - state.approx_loglik

    # uniform prior on [0,1]² stub
    log_accept = delta_l
    accepted = math.log(rng.uniform()) < log_accept

    if accepted:
        return True, MCMCState(
            tau=tau_star,
            partition=state.partition,
            edges=state.edges,
            store=store_star,
            approx_loglik=l_tilde_star,
            cfg=state.cfg,
        )
    return False, state


def run_embedding_sweep(
    state: MCMCState,
    *,
    rng: np.random.Generator,
) -> tuple[int, MCMCState]:
    """One full vertex sweep; returns accept count."""
    accepts = 0
    n = state.tau.shape[0]
    order = rng.permutation(n)
    for i in order:
        ok, state = propose_embedding_step(state, int(i), rng=rng)
        if ok:
            accepts += 1
    return accepts, state
