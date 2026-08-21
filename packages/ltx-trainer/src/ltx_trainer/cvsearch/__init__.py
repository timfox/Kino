"""CVSearch: cognitive visual search for HR MLLM perception (Li et al., ICML 2026)."""

from ltx_trainer.cvsearch.config import CVSearchConfig, SearchMode, adaptive_tree_depth
from ltx_trainer.cvsearch.layout import LIMITATIONS
from ltx_trainer.cvsearch.search import (
    BottomUpResult,
    bottom_up_search_layer,
    dynamic_threshold,
    expert_coverage_valid,
    information_sufficiency,
    node_priority,
    route_search_mode,
)
from ltx_trainer.cvsearch.sgap import (
    TreeNode,
    clustering_cost,
    overlap_penalty,
    select_optimal_k,
    silhouette_score,
    visual_complexity,
)
from ltx_trainer.cvsearch.pipeline import (
    bottom_up_demo,
    evaluation_demo,
    figure1_highlights,
    framework_card,
    routing_demo,
    sgap_demo,
    table_ablation,
    table_general_benchmarks,
    table_hr_benchmarks,
    table_throughput,
    table_visual_search_comparison,
)

__all__ = [
    "BottomUpResult",
    "CVSearchConfig",
    "LIMITATIONS",
    "SearchMode",
    "TreeNode",
    "adaptive_tree_depth",
    "bottom_up_demo",
    "bottom_up_search_layer",
    "clustering_cost",
    "dynamic_threshold",
    "evaluation_demo",
    "expert_coverage_valid",
    "figure1_highlights",
    "framework_card",
    "information_sufficiency",
    "node_priority",
    "overlap_penalty",
    "route_search_mode",
    "routing_demo",
    "select_optimal_k",
    "sgap_demo",
    "silhouette_score",
    "table_ablation",
    "table_general_benchmarks",
    "table_hr_benchmarks",
    "table_throughput",
    "table_visual_search_comparison",
    "visual_complexity",
]
