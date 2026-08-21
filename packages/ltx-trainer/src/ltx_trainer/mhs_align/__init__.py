"""MHS attribute-level LLM alignment stub (arXiv:2605.27025)."""

from ltx_trainer.mhs_align.config import MhsAlignConfig
from ltx_trainer.mhs_align.layout import (
    BEHAVIORAL_ATTRIBUTES,
    EVALUATIVE_ATTRIBUTES,
    LIMITATIONS,
    MHS_ATTRIBUTES,
    PIPELINE_STAGES,
)
from ltx_trainer.mhs_align.mock import evaluation_smoke
from ltx_trainer.mhs_align.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.mhs_align.ridge import confidence_weighted_features, ridge_reconstruction_smoke
from ltx_trainer.mhs_align.tables import headline_results

__all__ = [
    "BEHAVIORAL_ATTRIBUTES",
    "EVALUATIVE_ATTRIBUTES",
    "LIMITATIONS",
    "MHS_ATTRIBUTES",
    "MhsAlignConfig",
    "PIPELINE_STAGES",
    "benchmarks_bundle",
    "confidence_weighted_features",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "ridge_reconstruction_smoke",
]
