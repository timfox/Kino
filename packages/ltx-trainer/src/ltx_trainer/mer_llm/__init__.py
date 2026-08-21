"""MER-with-LLMs survey: multimodal emotion recognition via MLLMs (arXiv:2605.21239)."""

from ltx_trainer.mer_llm.catalog import methods_for_subtask, representative_methods, research_radar_brief
from ltx_trainer.mer_llm.challenges import three_challenges
from ltx_trainer.mer_llm.knowledge import mer_llm_knowledge_blob
from ltx_trainer.mer_llm.config import MerLlmConfig
from ltx_trainer.mer_llm.formulation import autoregressive_mer_step, mer_with_llms_response_schema
from ltx_trainer.mer_llm.layout import LIMITATIONS
from ltx_trainer.mer_llm.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    training_step_demo,
)
from ltx_trainer.mer_llm.scoring import explainable_mer_stub, paradigm_lift
from ltx_trainer.mer_llm.tables import (
    fig1b_paradigm_progress,
    table1_emotion_datasets_excerpt,
    table2_perceptual_mapping_excerpt,
    table5_quantitative_excerpt,
)
from ltx_trainer.mer_llm.taxonomy import five_subtasks, future_directions, taxonomy_branches

__all__ = [
    "LIMITATIONS",
    "MerLlmConfig",
    "autoregressive_mer_step",
    "benchmarks_bundle",
    "evaluation_demo",
    "explainable_mer_stub",
    "fig1b_paradigm_progress",
    "five_subtasks",
    "framework_card",
    "future_directions",
    "mer_llm_knowledge_blob",
    "mer_with_llms_response_schema",
    "methods_for_subtask",
    "paradigm_lift",
    "representative_methods",
    "research_radar_brief",
    "table1_emotion_datasets_excerpt",
    "table2_perceptual_mapping_excerpt",
    "table5_quantitative_excerpt",
    "taxonomy_branches",
    "three_challenges",
    "training_step_demo",
]
