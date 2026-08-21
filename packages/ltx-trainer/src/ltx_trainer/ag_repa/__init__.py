"""AG-REPA: causal layer selection for audio Flow Matching REPA (arXiv:2603.01006)."""

from ltx_trainer.ag_repa.ag_repa import AgRepaLossBreakdown, total_ag_repa_loss
from ltx_trainer.ag_repa.config import AgRepaConfig
from ltx_trainer.ag_repa.fog_a import FogAScore, fog_a_score, top_k_by_fog_a
from ltx_trainer.ag_repa.lasp import LayerScore, lasp_layer_score, top_k_layers
from ltx_trainer.ag_repa.mock import evaluation_smoke
from ltx_trainer.ag_repa.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
)
from ltx_trainer.ag_repa.tables import (
    table1_scd_dissociation,
    table2_alignment_strategies,
    table3_selection_targets,
)
from ltx_trainer.ag_repa.taxonomy import SelectionStrategy, STRATEGY_LABELS

__all__ = [
    "AgRepaConfig",
    "AgRepaLossBreakdown",
    "FogAScore",
    "LayerScore",
    "SelectionStrategy",
    "STRATEGY_LABELS",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "fog_a_score",
    "framework_card",
    "knowledge_card",
    "lasp_layer_score",
    "table1_scd_dissociation",
    "table2_alignment_strategies",
    "table3_selection_targets",
    "top_k_by_fog_a",
    "top_k_layers",
    "total_ag_repa_loss",
]
