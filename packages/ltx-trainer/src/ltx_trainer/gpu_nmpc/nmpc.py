"""GPU-NMPC loop with parametric re-solves (Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.gpu_nmpc.benchmarks import distillation_tray1_step, plate_average_step, plate_setpoint, scalar_plant_step
from ltx_trainer.gpu_nmpc.cholesky_cache import CholeskyCache
from ltx_trainer.gpu_nmpc.config import DistillationParams, GpuNmpcConfig, HeatedPlateParams, ResolveMode
from ltx_trainer.gpu_nmpc.ipm import lifted_ipm
from ltx_trainer.gpu_nmpc.transcription import transcribe_distillation, transcribe_scalar_ocp


@dataclass
class NmpcRunSummary:
    benchmark: str
    mode: ResolveMode
    steps: int
    total_setup_ms: float
    total_solve_ms: float
    total_ms: float
    final_state: float
    reused_symbolic_all_but_first: bool


def run_nmpc_loop(
    *,
    benchmark: str,
    cfg: GpuNmpcConfig,
    mode: ResolveMode = "warmstart_param_update",
    steps: int | None = None,
    seed: int = 42,
) -> NmpcRunSummary:
    steps = steps or cfg.demo_nmpc_steps
    cache: CholeskyCache | None = None
    z: np.ndarray | None = None
    total_setup = 0.0
    total_solve = 0.0
    reused_after_first = True

    if benchmark == "distillation":
        params_cfg = cfg.distillation
        ocp = transcribe_distillation(n_trays=cfg.demo_n_trays, horizon_nodes=cfg.demo_horizon_nodes)
        state = 0.5
        dt = params_cfg.dt
        for k in range(steps):
            p = np.array([state, params_cfg.x_setpoint, params_cfg.u_setpoint, params_cfg.feed_conc])
            res, cache = lifted_ipm(ocp, p, z_init=z, cache=cache, cfg=cfg, mode=mode, seed=seed + k)
            total_setup += res.setup_ms
            total_solve += res.solve_ms
            if k > 0 and not res.reused_symbolic:
                reused_after_first = False
            z = res.primal
            u_opt = float(np.clip(params_cfg.u_setpoint + 0.1 * res.primal[0], 1.0, 5.0))
            state = distillation_tray1_step(state, u_opt, params=params_cfg, dt=dt)
        final = state
    elif benchmark == "pde_plate":
        params_cfg = cfg.plate
        ocp = transcribe_scalar_ocp(horizon_nodes=cfg.demo_horizon_nodes, n_state=1, n_control=1)
        state = 50.0
        t = 0.0
        dt = params_cfg.dt
        for k in range(steps):
            t_set = plate_setpoint(t, params_cfg.t_final)
            p = np.array([state, t_set])
            res, cache = lifted_ipm(ocp, p, z_init=z, cache=cache, cfg=cfg, mode=mode, seed=seed + 100 + k)
            total_setup += res.setup_ms
            total_solve += res.solve_ms
            if k > 0 and not res.reused_symbolic:
                reused_after_first = False
            z = res.primal
            q = float(np.clip(50.0 + res.primal[-1], 50.0, 200.0))
            state = plate_average_step(state, q, params=params_cfg, dt=dt, t_set=t_set)
            t += dt
        final = state
    else:
        ocp = transcribe_scalar_ocp(horizon_nodes=cfg.demo_horizon_nodes)
        state = 1.0
        dt = 1.0
        for k in range(steps):
            p = np.array([state, float(k)])
            res, cache = lifted_ipm(ocp, p, z_init=z, cache=cache, cfg=cfg, mode=mode, seed=seed + k)
            total_setup += res.setup_ms
            total_solve += res.solve_ms
            if k > 0 and not res.reused_symbolic:
                reused_after_first = False
            z = res.primal
            u_opt = float(np.clip(2.0 + 0.05 * res.primal[0], 1.0, 3.0))
            state = scalar_plant_step(state, u_opt, dt)
        final = state

    total = total_setup + total_solve
    return NmpcRunSummary(
        benchmark=benchmark,
        mode=mode,
        steps=steps,
        total_setup_ms=total_setup,
        total_solve_ms=total_solve,
        total_ms=total,
        final_state=final,
        reused_symbolic_all_but_first=reused_after_first,
    )
