"""Augment Engineering multi-tool orchestration stub (arXiv:2605.26146)."""

from ltx_trainer.augment_engineering.config import AugmentEngConfig
from ltx_trainer.augment_engineering.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.augment_engineering.metrics import (
    coverage_breadth,
    orchestration_overhead,
    transfer_velocity_hours,
)
from ltx_trainer.augment_engineering.mock import evaluation_smoke
from ltx_trainer.augment_engineering.phases import orchestration_patterns, phase_catalog
from ltx_trainer.augment_engineering.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.augment_engineering.rubric import classify_prompt_level, rubric_levels
from ltx_trainer.augment_engineering.stats import cochran_armitage_trend, wrights_law_fit
from ltx_trainer.augment_engineering.tables import headline_results

__all__ = [
    "LIMITATIONS",
    "PIPELINE_STAGES",
    "AugmentEngConfig",
    "benchmarks_bundle",
    "classify_prompt_level",
    "cochran_armitage_trend",
    "coverage_breadth",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "orchestration_overhead",
    "orchestration_patterns",
    "phase_catalog",
    "rubric_levels",
    "transfer_velocity_hours",
    "wrights_law_fit",
]
