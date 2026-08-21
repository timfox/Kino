"""EvoGS configuration (arXiv:2606.07179)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RefinementMode(str, Enum):
    """Parent-child refinement construction (Figure 4)."""

    INDEPENDENT = "option_a"
    INDEP_RESIDUAL = "option_b"
    SYMMETRIC = "option_c"
    ASYMMETRIC = "option_d"


@dataclass(frozen=True)
class EvoGSParams:
    num_levels: int = 4
    downsample_factors: tuple[int, ...] = (8, 4, 2, 1)
    base_iters: int = 30_000
    refine_iters: int = 30_000
    densify_interval: int = 3_000
    densify_until: int = 15_000
    lambda_dssim: float = 0.2
    grad_split_threshold: float = 2e-4
    opacity_ghost_threshold: float = 0.005
    refinement_mode: RefinementMode = RefinementMode.ASYMMETRIC
    alpha_groups: int = 5
    param_dim: int = 59


@dataclass(frozen=True)
class EvoGSConfig:
    paper_arxiv: str = "arXiv:2606.07179"
    paper_title: str = (
        "EvoGS: Constructing Continuous-Layered Gaussian Splatting with Evolution Tree "
        "for Scalable 3D Streaming"
    )
    datasets: tuple[str, ...] = ("Blender", "Mip-NeRF360", "Tanks&Temples", "Deep Blending")
    baselines: tuple[str, ...] = ("Monolithic", "Single", "L3GS", "LapisGS", "EvoGS (Sym.)", "EvoGS")
    packages: tuple[str, ...] = ("evogs",)
    params: EvoGSParams = field(default_factory=EvoGSParams)
