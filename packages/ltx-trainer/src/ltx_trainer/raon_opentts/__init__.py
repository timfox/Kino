"""Raon-OpenTTS — open pool, core filter, DiT TTS, robust eval stub (arXiv:2605.20830)."""

from ltx_trainer.raon_opentts.config import RaonOpenTtsConfig
from ltx_trainer.raon_opentts.filtering import keep_top_percentile_by_combined_rank, mean_rank_scores
from ltx_trainer.raon_opentts.layout import LIMITATIONS
from ltx_trainer.raon_opentts.mock import evaluation_smoke
from ltx_trainer.raon_opentts.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    filtering_thresholds_card,
    framework_card,
    headline_results,
    table1_seed_tts_eval_subset,
    table2_pool_composition_excerpt,
    table3_filtering_ablation_excerpt,
    table4_core_retention_excerpt,
    table5_cv3_excerpt,
    table6_raon_eval_overall,
)

__all__ = [
    "LIMITATIONS",
    "RaonOpenTtsConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "filtering_thresholds_card",
    "framework_card",
    "headline_results",
    "keep_top_percentile_by_combined_rank",
    "mean_rank_scores",
    "table1_seed_tts_eval_subset",
    "table2_pool_composition_excerpt",
    "table3_filtering_ablation_excerpt",
    "table4_core_retention_excerpt",
    "table5_cv3_excerpt",
    "table6_raon_eval_overall",
]
