"""Lifted-KKT parametric interior-point step (Algorithms 1–2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.gpu_nmpc.cholesky_cache import CholeskyCache, sym_cholesky, solve_with_cache
from ltx_trainer.gpu_nmpc.config import GpuNmpcConfig, ResolveMode
from ltx_trainer.gpu_nmpc.transcription import TranscribedOcp, build_condensed_spd


@dataclass
class IpSolveResult:
    primal: np.ndarray
    setup_ms: float
    solve_ms: float
    newton_steps: int
    barrier_mu: float
    reused_symbolic: bool


def _mode_flags(mode: ResolveMode) -> tuple[bool, bool]:
    warm = mode in ("warmstart", "warmstart_param_update")
    pu = mode in ("param_update", "warmstart_param_update")
    return warm, pu


def lifted_ipm(
    ocp: TranscribedOcp,
    params: np.ndarray,
    *,
    z_init: np.ndarray | None,
    cache: CholeskyCache | None,
    cfg: GpuNmpcConfig,
    mode: ResolveMode = "warmstart_param_update",
    seed: int = 0,
) -> tuple[IpSolveResult, CholeskyCache]:
    """One NMPC OCP solve with optional warmstart + parameter update."""
    warm, pu = _mode_flags(mode)
    if cache is None:
        cache = CholeskyCache()

    setup_ms = 0.0
    reused_symbolic = cache.symbolic_done

    if not pu and cache.symbolic_done:
        cache = CholeskyCache()
        reused_symbolic = False

    k = build_condensed_spd(ocp, seed=seed, reg=cfg.inertia_reg)
    rhs = np.random.default_rng(seed + 1).standard_normal(ocp.n_primal) * 0.1
    if params.size:
        rhs[0] += float(params[0]) * 0.01

    if not cache.symbolic_done:
        sym = sym_cholesky(k, cost_unit_ms=cfg.symbolic_cost_units)
        cache.permutation = sym.permutation
        cache.sparsity_pairs = sym.sparsity_pairs
        cache.elimination_tree_depth = sym.elimination_tree_depth
        cache.symbolic_done = True
        cache.symbolic_ms += sym.symbolic_ms
        setup_ms += sym.symbolic_ms
    elif warm and not pu:
        setup_ms += cfg.rebuild_cost_units

    n_steps = 3 if warm else 6
    solve_ms = 0.0
    z = np.zeros(ocp.n_primal) if z_init is None or not warm else z_init.copy()
    mu = cfg.barrier_mu0
    for step in range(n_steps):
        solve_ms += cfg.numeric_factor_cost_units + cfg.newton_step_cost_units
        delta = solve_with_cache(k, -rhs * (0.5 ** step), cache, cost_unit_ms=cfg.numeric_factor_cost_units)
        z = z + 0.5 * delta
        mu *= 0.5

    return (
        IpSolveResult(
            primal=z,
            setup_ms=setup_ms,
            solve_ms=solve_ms,
            newton_steps=n_steps,
            barrier_mu=mu,
            reused_symbolic=reused_symbolic,
        ),
        cache,
    )
