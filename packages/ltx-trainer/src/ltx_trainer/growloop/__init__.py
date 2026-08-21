"""GrowLoop: self-evolving conversation evaluation (arXiv:2605.28882)."""

from ltx_trainer.growloop.config import EvaluationZone, GrowLoopConfig, GrowStage, RubricTrack
from ltx_trainer.growloop.case_store import case_discriminability, load_cases
from ltx_trainer.growloop.csp import CSP_GROUPS, csp_field_registry, csp_pool_sources
from ltx_trainer.growloop.dual_loop_runner import dual_loop_runner_smoke, run_dual_loop
from ltx_trainer.growloop.rubric_editor import default_rubric, refine_rubric, rubric_editor_smoke
from ltx_trainer.growloop.dual_loop import DUAL_LOOP_TRIGGERS, TriggerCell, anchor_drift_ok, dual_loop_smoke, select_trigger
from ltx_trainer.growloop.gates import evaluate_gates, gate_registry, gates_smoke
from ltx_trainer.growloop.heuristic import (
    HeuristicState,
    cascaded_final_score,
    heuristic_learning_step,
    heuristic_smoke,
    run_heuristic_learning,
    stage_pipeline,
)
from ltx_trainer.growloop.layout import LIMITATIONS
from ltx_trainer.growloop.metrics import (
    cascaded_score,
    cliffs_delta,
    kendall_tau,
    metrics_smoke,
    pair_accuracy,
    spearman_rho,
    tie_aware_accuracy,
)
from ltx_trainer.growloop.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_vii_cross_model,
    table_viii_baselines,
    table_ix_capabilities,
    table_xi_tier_profile,
    table_xiv_feedback_ablation,
)
from ltx_trainer.growloop.zones import partition_zones, zone_aware_agreement, zones_smoke

__all__ = [
    "CSP_GROUPS",
    "DUAL_LOOP_TRIGGERS",
    "EvaluationZone",
    "GrowLoopConfig",
    "GrowStage",
    "HeuristicState",
    "LIMITATIONS",
    "RubricTrack",
    "TriggerCell",
    "anchor_drift_ok",
    "case_discriminability",
    "benchmarks_bundle",
    "cascaded_final_score",
    "cascaded_score",
    "cliffs_delta",
    "csp_field_registry",
    "csp_pool_sources",
    "default_rubric",
    "dual_loop_runner_smoke",
    "dual_loop_smoke",
    "evaluate_gates",
    "evaluation_demo",
    "framework_card",
    "gate_registry",
    "gates_smoke",
    "headline_results",
    "heuristic_learning_step",
    "heuristic_smoke",
    "kendall_tau",
    "load_cases",
    "metrics_smoke",
    "pair_accuracy",
    "partition_zones",
    "pipeline_demo",
    "refine_rubric",
    "rubric_editor_smoke",
    "run_dual_loop",
    "run_heuristic_learning",
    "select_trigger",
    "spearman_rho",
    "stage_pipeline",
    "table_vii_cross_model",
    "table_viii_baselines",
    "table_ix_capabilities",
    "table_xi_tier_profile",
    "table_xiv_feedback_ablation",
    "tie_aware_accuracy",
    "zone_aware_agreement",
    "zones_smoke",
]
