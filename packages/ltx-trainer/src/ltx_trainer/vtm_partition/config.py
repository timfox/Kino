"""VTM partition acceleration survey + RL stub (arXiv:2605.21526)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VTMPartitionConfig:
    paper_arxiv: str = "arXiv:2605.21526"
    reference_vtm: str = "VTM-18.0"
    quantization_parameters: tuple[int, ...] = (22, 27, 32, 37)
    split_modes: tuple[str, ...] = ("NS", "QT", "BTH", "BTV", "TTH", "TTV")
    rl_feature_dim: int = 14
    rl_top_n_choices: tuple[int, ...] = (2, 3, 4, 5)
    rl_q_threshold: float = 0.15
    loss_alphas: tuple[float, float, float] = (1.0, 1.0, 0.5)
    discount_gamma: float = 1.0
    training_trajectory_sizes: tuple[str, ...] = ("32x32", "16x16", "32x16", "16x32", "8x32", "32x8")
    vtm_versions_surveyed: tuple[str, ...] = (
        "VTM-3.0",
        "VTM-5.0",
        "VTM-6.0",
        "VTM-7.0",
        "VTM-8.0",
        "VTM-9.0",
        "VTM-10.2",
        "VTM-18.0",
        "VTM-23.11",
    )
