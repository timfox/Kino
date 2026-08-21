"""AdaMaG — Adaptive Manifold Guidance for flow/diffusion sampling."""

from ltx_trainer.adamag.config import AdamagConfig
from ltx_trainer.adamag.guidance import (
    adamag_guidance_residual,
    adamag_velocity,
    cfg_velocity,
    decompose_guidance,
    omega_schedule,
    score_normal_direction,
)
from ltx_trainer.adamag.layout import CONSERVATION_EQ5, LIMITATIONS, PIPELINE_STEPS
from ltx_trainer.adamag.mock import evaluation_smoke
from ltx_trainer.adamag.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
)
from ltx_trainer.adamag.tables import (
    table1_high_guidance,
    table1_optimal_guidance,
    table4_beta_ablation,
    table5_gamma_ablation,
)

__all__ = [
    "CONSERVATION_EQ5",
    "LIMITATIONS",
    "PIPELINE_STEPS",
    "AdamagConfig",
    "adamag_guidance_residual",
    "adamag_velocity",
    "benchmarks_bundle",
    "cfg_velocity",
    "decompose_guidance",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "omega_schedule",
    "score_normal_direction",
    "table1_high_guidance",
    "table1_optimal_guidance",
    "table4_beta_ablation",
    "table5_gamma_ablation",
]
