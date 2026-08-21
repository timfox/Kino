"""GPU-NMPC configuration (arXiv:2606.04725)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResolveMode = Literal["base", "warmstart", "param_update", "warmstart_param_update"]


@dataclass(frozen=True)
class DistillationParams:
    n_trays: int = 32
    feed_tray: int = 17
    distillate_flow: float = 0.2
    feed_flow: float = 0.4
    relative_volatility: float = 1.6
    x_setpoint: float = 0.8958
    u_setpoint: float = 2.51459
    feed_conc: float = 0.5
    ac: float = 0.5
    at: float = 0.25
    ar: float = 1.0
    t_final: float = 300.0
    horizon: float = 180.0
    dt: float = 2.0


@dataclass(frozen=True)
class HeatedPlateParams:
    thermal_conductivity: float = 230.0
    length: float = 0.1
    boundary_temp: float = 50.0
    grid_points: int = 21
    rho_penalty: float = 1.0
    heat_loss: float = 1e-8
    t_final: float = 3.0
    horizon: float = 1.0
    dt: float = 0.08


@dataclass
class GpuNmpcConfig:
    paper_arxiv: str = "arXiv:2606.04725"
    packages: tuple[str, ...] = ("InfiniteOpt.jl", "InfiniteExaModels.jl", "MadNLP.jl", "cuDSS")
    distillation: DistillationParams = field(default_factory=DistillationParams)
    plate: HeatedPlateParams = field(default_factory=HeatedPlateParams)
    inner_tol: float = 1e-4
    outer_tol: float = 1e-3
    barrier_mu0: float = 1.0
    lifted_gamma: float = 1e-8
    inertia_reg: float = 1e-6
    demo_horizon_nodes: int = 8
    demo_n_trays: int = 4
    demo_nmpc_steps: int = 5
    symbolic_cost_units: float = 1.0
    rebuild_cost_units: float = 0.85
    numeric_factor_cost_units: float = 0.02
    newton_step_cost_units: float = 0.01
