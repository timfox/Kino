"""UniSRM unified speech reward model stub (arXiv:2605.23261)."""

from ltx_trainer.unisrm.config import UnisrmConfig
from ltx_trainer.unisrm.layout import LIMITATIONS
from ltx_trainer.unisrm.mock import evaluation_smoke
from ltx_trainer.unisrm.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table_1_unisrm_bench,
    table_2_ablation,
    table_3_task1_dimensions,
    table_4_task2_pcc,
    table_7_cross_dataset,
    table_9_dataset_stats,
)
from ltx_trainer.unisrm.rewards import (
    accuracy_reward_mos,
    accuracy_reward_pairwise,
    combined_reward,
    format_reward,
    grpo_advantages,
    reasoning_consistent_mos,
    reasoning_consistent_pairwise,
)

__all__ = [
    "LIMITATIONS",
    "UnisrmConfig",
    "accuracy_reward_mos",
    "accuracy_reward_pairwise",
    "benchmarks_bundle",
    "combined_reward",
    "evaluation_demo",
    "evaluation_smoke",
    "format_reward",
    "framework_card",
    "grpo_advantages",
    "headline_results",
    "reasoning_consistent_mos",
    "reasoning_consistent_pairwise",
    "table_1_unisrm_bench",
    "table_2_ablation",
    "table_3_task1_dimensions",
    "table_4_task2_pcc",
    "table_7_cross_dataset",
    "table_9_dataset_stats",
]
