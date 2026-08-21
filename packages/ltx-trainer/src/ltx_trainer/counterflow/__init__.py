"""CounterFlow — counterfactual video foley via two-phase sampling (arXiv:2605.18916)."""

from ltx_trainer.counterflow.config import CounterFlowConfig
from ltx_trainer.counterflow.guidance import (
    euler_sample_counterflow,
    phase1_velocity,
    phase2_velocity,
    vanilla_cfg_velocity,
)
from ltx_trainer.counterflow.layout import LIMITATIONS
from ltx_trainer.counterflow.metrics import delta_flam, p_flam_max_frame, positive_delta_flam_ratio
from ltx_trainer.counterflow.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    figure3_ntrans_sweep,
    framework_card,
    pipeline_demo,
    table_i_main_comparison,
    table_ii_ablations,
)

__all__ = [
    "CounterFlowConfig",
    "LIMITATIONS",
    "benchmarks_bundle",
    "delta_flam",
    "evaluation_demo",
    "euler_sample_counterflow",
    "figure3_ntrans_sweep",
    "framework_card",
    "p_flam_max_frame",
    "phase1_velocity",
    "phase2_velocity",
    "pipeline_demo",
    "positive_delta_flam_ratio",
    "table_i_main_comparison",
    "table_ii_ablations",
    "vanilla_cfg_velocity",
]
