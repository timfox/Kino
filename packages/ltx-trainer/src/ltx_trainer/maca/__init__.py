"""MACA — Multi-Agent Coordination Adaptation (Structure-Guided Orchestration, arXiv:2605.25746)."""

from ltx_trainer.maca.config import MACAConfig
from ltx_trainer.maca.embeddings import AgentProfile, budget_temperature, cosine, embed_text
from ltx_trainer.maca.layout import architecture_layout, paper_limitations
from ltx_trainer.maca.orchestration import (
    OrchestrationPolicy,
    compare_with_without_graphspec,
    grpo_update_demo,
    sample_trajectory,
)
from ltx_trainer.maca.pipeline import (
    evaluation_demo,
    framework_card,
    graphspec_demo,
    table_ablation_arc_gsm,
    table_main_results_llama31_8b,
    table_sensitivity,
)
from ltx_trainer.maca.prior import GraphSpec, InteractionMLP, agent_relevance_scores, build_graphspec

__all__ = [
    "AgentProfile",
    "GraphSpec",
    "InteractionMLP",
    "MACAConfig",
    "OrchestrationPolicy",
    "agent_relevance_scores",
    "architecture_layout",
    "budget_temperature",
    "build_graphspec",
    "compare_with_without_graphspec",
    "cosine",
    "embed_text",
    "evaluation_demo",
    "framework_card",
    "graphspec_demo",
    "grpo_update_demo",
    "paper_limitations",
    "sample_trajectory",
    "table_ablation_arc_gsm",
    "table_main_results_llama31_8b",
    "table_sensitivity",
]

