"""VISA — Visual Information Strengthened Audio-Reasoning (arXiv:2606.07264)."""

from ltx_trainer.visa.config import VisaConfig
from ltx_trainer.visa.eval import eval_smoke, pipeline_demo
from ltx_trainer.visa.features import multimodal_feature_bundle
from ltx_trainer.visa.fold import annotate_audio_save_data
from ltx_trainer.visa.ltx_plan import ltx_integration_plan
from ltx_trainer.visa.metrics import aggregate_accuracy, mc_accuracy, rubrics_components
from ltx_trainer.visa.mock import evaluation_smoke
from ltx_trainer.visa.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_mmar_modality,
    table2_subcategories,
    table3_routing_taxonomy,
    table4_leaderboard,
)
from ltx_trainer.visa.routing import disagree_then_route, routing_demo
from ltx_trainer.visa.taxonomy import category_registry, lookup_category, routing_strategy_counts
from ltx_trainer.visa.voting import ensemble_vote_inference, majority_vote, model_vote_single

__all__ = [
    "VisaConfig",
    "aggregate_accuracy",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "category_registry",
    "disagree_then_route",
    "ensemble_vote_inference",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "lookup_category",
    "ltx_integration_plan",
    "majority_vote",
    "mc_accuracy",
    "model_vote_single",
    "multimodal_feature_bundle",
    "pipeline_demo",
    "routing_demo",
    "routing_strategy_counts",
    "rubrics_components",
    "table1_mmar_modality",
    "table2_subcategories",
    "table3_routing_taxonomy",
    "table4_leaderboard",
]
