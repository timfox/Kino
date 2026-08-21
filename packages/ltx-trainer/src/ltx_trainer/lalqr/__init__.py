"""Latent Activation LQR (LA-LQR) for T2V steering — arXiv:2606.04775."""

from ltx_trainer.lalqr.config import LalqrConfig, LQRWeights
from ltx_trainer.lalqr.contrastive_prompts import list_categories, pair_for
from ltx_trainer.lalqr.infer_bridge import (
    apply_lalqr_to_prompt_contexts,
    infer_lalqr_enabled,
    ltx_integration_notes,
    steer_text_context_numpy,
    steer_text_context_torch,
)
from ltx_trainer.lalqr.lqr import la_lqr_text_control, solve_ltv_lqr
from ltx_trainer.lalqr.metrics import (
    average_violation,
    projection_calibrated_setpoint,
    raw_latent_tracking_bound,
    table1_t2vsafetybench,
    table1_vbench_subject,
    table2_safesora,
)
from ltx_trainer.lalqr.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    table1_summary,
    table2_summary,
)
from ltx_trainer.lalqr.setpoints import (
    latent_feature_direction,
    latent_feature_strength,
    llfs_setpoint,
    tracking_error,
)
from ltx_trainer.lalqr.steering import (
    build_latent_controller,
    run_lalqr_smoke,
    steer_latent_chain,
    synthesize_contrastive_activations,
)
from ltx_trainer.lalqr.subspace import (
    captured_energy_fraction,
    contrastive_rows,
    mean_contrastive_vector,
    project_activation,
    randomized_svd_basis,
)

__all__ = [
    "LQRWeights",
    "LalqrConfig",
    "apply_lalqr_to_prompt_contexts",
    "average_violation",
    "benchmark_manifest",
    "build_latent_controller",
    "captured_energy_fraction",
    "contrastive_rows",
    "evaluation_demo",
    "framework_card",
    "infer_lalqr_enabled",
    "la_lqr_text_control",
    "latent_feature_direction",
    "latent_feature_strength",
    "list_categories",
    "llfs_setpoint",
    "ltx_integration_notes",
    "mean_contrastive_vector",
    "pair_for",
    "paper_limitations",
    "project_activation",
    "projection_calibrated_setpoint",
    "randomized_svd_basis",
    "raw_latent_tracking_bound",
    "run_lalqr_smoke",
    "solve_ltv_lqr",
    "steer_latent_chain",
    "steer_text_context_numpy",
    "steer_text_context_torch",
    "synthesize_contrastive_activations",
    "table1_summary",
    "table1_t2vsafetybench",
    "table1_vbench_subject",
    "table2_safesora",
    "table2_summary",
    "tracking_error",
]
