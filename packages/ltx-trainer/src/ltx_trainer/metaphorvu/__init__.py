"""MetaphorVU: metaphorical video understanding (Li et al., arXiv:2605.25488)."""

from ltx_trainer.metaphorvu.boost import build_demo_graph, metaphor_boost
from ltx_trainer.metaphorvu.config import MetaphorVUConfig
from ltx_trainer.metaphorvu.kg import MetaphorKnowledgeGraph, top_z_references
from ltx_trainer.metaphorvu.taxonomy import (
    METAPHOR_TYPES,
    benchmark_type_stats,
    taxonomy_card,
)
from ltx_trainer.metaphorvu.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_metaphorboost,
    table_error_analysis,
    table_gemini3_by_type,
    table_hyperparameter_query,
    table_overall_results,
    training_step_demo,
)

__all__ = [
    "METAPHOR_TYPES",
    "MetaphorKnowledgeGraph",
    "MetaphorVUConfig",
    "benchmark_type_stats",
    "build_demo_graph",
    "evaluation_demo",
    "framework_card",
    "metaphor_boost",
    "table_ablation_metaphorboost",
    "table_error_analysis",
    "table_gemini3_by_type",
    "table_hyperparameter_query",
    "table_overall_results",
    "taxonomy_card",
    "top_z_references",
    "training_step_demo",
]
