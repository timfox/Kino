"""SLE κ neural-network prediction config (Shrotri & Margarint, arXiv:2606.02682)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoewnerSimConfig:
    """Deterministic ξ(t)=c√t and stochastic SLEκ simulation settings."""

    c_min: float = 0.0
    c_max: float = 3.0
    kappa_min: float = 0.0
    kappa_max: float = 8.0
    t_start: float = 0.1
    t_end: float = 1.0
    n_steps: int = 100
    n_trajectories: int = 10_000
    train_fraction: float = 0.8


@dataclass(frozen=True)
class NNArchitecture:
    """Paper architectures (Sec. 2.3, 3.3)."""

    deterministic_hidden: tuple[int, ...] = (64, 64)
    sle_hidden: tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.2


@dataclass
class SleNNConfig:
    paper_arxiv: str = "2606.02682"
    sim: LoewnerSimConfig = field(default_factory=LoewnerSimConfig)
    nn: NNArchitecture = field(default_factory=NNArchitecture)
    test_mse_deterministic_c: float = 0.00264
    test_mse_sle_same_noise: float = 0.345
    test_mse_sle_different_noise: float = 3.98
    test_mse_sle_trace_fixed_bm: float = 0.194
