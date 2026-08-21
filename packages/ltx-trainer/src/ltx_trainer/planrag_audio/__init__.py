"""PlanRAG-Audio — planning-based RAG for long-form audio (arXiv:2605.20414)."""

from ltx_trainer.planrag_audio.config import PlanRagAudioConfig
from ltx_trainer.planrag_audio.fusion import TimeSegment, fuse_nearest_midpoint, temporal_overlap
from ltx_trainer.planrag_audio.layout import LIMITATIONS
from ltx_trainer.planrag_audio.mock import evaluation_smoke, example_plan_dict
from ltx_trainer.planrag_audio.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_audio_database_example_rows,
    table2_task_query_examples,
    table3_model_configs,
    table19_keyword_vs_vector_retrieval,
    table4_mcqa_llm_tokens_60min,
    table5_speaker_count_event_order,
    table6_speaker_constrained_mcqa,
    table7_error_decomposition_rows,
)
from ltx_trainer.planrag_audio.sql_toy import merged_sql_sketch

__all__ = [
    "LIMITATIONS",
    "PlanRagAudioConfig",
    "TimeSegment",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "example_plan_dict",
    "framework_card",
    "fuse_nearest_midpoint",
    "headline_results",
    "merged_sql_sketch",
    "table1_audio_database_example_rows",
    "table2_task_query_examples",
    "table3_model_configs",
    "table19_keyword_vs_vector_retrieval",
    "table4_mcqa_llm_tokens_60min",
    "table5_speaker_count_event_order",
    "table6_speaker_constrained_mcqa",
    "table7_error_decomposition_rows",
    "temporal_overlap",
]
