"""Dyna-Pruner configuration (arXiv:2606.15346)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DynaPrunerConfig:
    paper_arxiv: str = "2606.15346"
    paper_title: str = "Dyna-Pruner: Input-Adaptive Data–Model Co-Pruning"
    authors: tuple[str, ...] = (
        "Fuyan Zhang",
        "Yuqi Li",
        "Qing Xu",
        "Yingli Tian",
        "Edmond S.L. Ho",
    )

    # Default sparsity targets (Table III)
    data_sparsity_sd: float = 0.7
    model_sparsity_sw: float = 0.7

    # Unified loss weights (Eq. 4)
    lambda_d: float = 1e-3
    lambda_w: float = 1e-3

    # Mask generator / STE
    ste_threshold: float = 0.5
    receptive_field: int = 3

    backbones: tuple[str, ...] = ("SimVP", "ConvLSTM", "TAU")
    datasets: tuple[str, ...] = ("WeatherBench", "SEVIR", "TaxiBJ")
    baselines: tuple[str, ...] = ("Dense", "MP")

    # Reported efficiency (Table I, Jetson AGX Orin)
    typical_gflops_reduction_pct: float = 74.9
    typical_latency_reduction_pct: float = 60.3
    typical_mse_increase_pct: float = 1.3


@dataclass
class SparsityBudget:
    """Per-sample sparsity budget for data and model masks."""

    sd: float = 0.7
    sw: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 <= self.sd <= 1.0 or not 0.0 <= self.sw <= 1.0:
            raise ValueError("sd and sw must be in [0, 1]")
