"""SEABAD — Southeast Asian Bird Activity Detection (arXiv:2605.20853)."""

from ltx_trainer.seabad.balancing import gini_coefficient, gini_reduction_pct, per_species_base_allocation, salience_score
from ltx_trainer.seabad.config import SeabadConfig, SundalandCoverage
from ltx_trainer.seabad.dedup import is_exact_duplicate, mel_embedding_mean_std
from ltx_trainer.seabad.layout import LIMITATIONS
from ltx_trainer.seabad.mock import evaluation_smoke
from ltx_trainer.seabad.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    positive_pipeline_stages,
    table1_dataset_comparison,
    table2_negative_sources,
    table3_dataset_statistics,
    table4_geographic_distribution,
    table5_baseline_validation,
)
from ltx_trainer.seabad.segment import rank_windows_by_rms, window_rms

__all__ = [
    "LIMITATIONS",
    "SeabadConfig",
    "SundalandCoverage",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "gini_coefficient",
    "gini_reduction_pct",
    "headline_results",
    "is_exact_duplicate",
    "mel_embedding_mean_std",
    "per_species_base_allocation",
    "positive_pipeline_stages",
    "rank_windows_by_rms",
    "salience_score",
    "table1_dataset_comparison",
    "table2_negative_sources",
    "table3_dataset_statistics",
    "table4_geographic_distribution",
    "table5_baseline_validation",
    "window_rms",
]
