"""Multimodal pathos analysis stub (arXiv:2605.22732)."""

from ltx_trainer.pathos_mm.config import PathosMmConfig
from ltx_trainer.pathos_mm.layout import LIMITATIONS
from ltx_trainer.pathos_mm.metrics import spearman_rho, trust_pathos_in_range
from ltx_trainer.pathos_mm.mock import evaluation_smoke
from ltx_trainer.pathos_mm.pipeline import (
    banaszak_rhetoric_distribution,
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_russell_weights,
    table2_emo_db_gemini_match,
    table3_banaszak_descriptive,
    table4_spearman_correlations,
)
from ltx_trainer.pathos_mm.russell import (
    E2V_CLASSES,
    E2V_RUSSELL_WEIGHTS,
    russell_arousal_valence,
)

__all__ = [
    "LIMITATIONS",
    "PathosMmConfig",
    "E2V_CLASSES",
    "E2V_RUSSELL_WEIGHTS",
    "banaszak_rhetoric_distribution",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "russell_arousal_valence",
    "spearman_rho",
    "table1_russell_weights",
    "table2_emo_db_gemini_match",
    "table3_banaszak_descriptive",
    "table4_spearman_correlations",
    "trust_pathos_in_range",
]
