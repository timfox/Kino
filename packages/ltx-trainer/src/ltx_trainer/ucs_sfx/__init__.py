"""UCS SFX dataset unification (Beck & Lerch, DAFx26 / arXiv:2606.05571)."""

from ltx_trainer.ucs_sfx.cascade import UcsMatch, classify_tag, classify_tags, normalize_tag
from ltx_trainer.ucs_sfx.conflict import resolve_file_categories
from ltx_trainer.ucs_sfx.config import UcsSfxConfig
from ltx_trainer.ucs_sfx.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.ucs_sfx.mock import evaluation_smoke
from ltx_trainer.ucs_sfx.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_conversion,
    table2_envsound_sources,
    table3_benchmark_original,
    table4_envsound_benchmark,
)
from ltx_trainer.ucs_sfx.split import composite_key, split_distribution_correlation, stratified_split_indices

__all__ = [
    "UcsMatch",
    "UcsSfxConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "classify_tag",
    "classify_tags",
    "composite_key",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "normalize_tag",
    "pipeline_demo",
    "pipeline_demo_export",
    "resolve_file_categories",
    "split_distribution_correlation",
    "stratified_split_indices",
    "table1_conversion",
    "table2_envsound_sources",
    "table3_benchmark_original",
    "table4_envsound_benchmark",
]

from ltx_trainer.ucs_sfx.fold import annotate_audio_save_data  # noqa: E402
