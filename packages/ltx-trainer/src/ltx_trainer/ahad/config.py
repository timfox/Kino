"""AHAD config (arXiv:2606.05369)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AHADParams:
    subspace_dim: int = 8
    shift_radius: int = 1
    sigma_shift: float = 1.0
    sstv_scale: float = 1e-3
    sstv_weights: tuple[float, float, float] = (2e-3, 1e-3, 2e-3)
    lambda1: float = 0.1
    lambda2: float = 0.5
    d2cm_strength: float = 8e2
    learning_rate: float = 0.05
    iterations: int = 25


@dataclass(frozen=True)
class AHADConfig:
    paper_arxiv: str = "arXiv:2606.05369"
    datasets: tuple[str, ...] = ("Airport I", "Airport II", "MUUFL", "Urban I", "Urban II")
    had_baselines: tuple[str, ...] = (
        "LRASR",
        "GTVLRR",
        "LARTVAD",
        "SuperRPCA",
        "RGAE",
        "BockNet",
        "PUUNet",
        "GTHAD",
        "NL2Net",
        "OTAD",
    )
    params: AHADParams = field(default_factory=AHADParams)
    packages: tuple[str, ...] = ("hsi_ops", "regularizers", "criterion", "metrics", "simulation")
