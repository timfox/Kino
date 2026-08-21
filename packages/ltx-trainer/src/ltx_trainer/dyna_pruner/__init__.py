"""Dyna-Pruner: input-adaptive data–model co-pruning (arXiv:2606.15346)."""

from ltx_trainer.dyna_pruner.config import DynaPrunerConfig, SparsityBudget
from ltx_trainer.dyna_pruner.layout import LIMITATIONS
from ltx_trainer.dyna_pruner.loss import mse_task_loss, total_loss
from ltx_trainer.dyna_pruner.ltx_plan import (
    energy_aware_training_note,
    gopex_env_exports,
    ltx_inference_plan,
    ltx_prep_plan,
)
from ltx_trainer.dyna_pruner.masks import (
    apply_data_mask,
    hard_mask_ste,
    importance_from_temporal_variance,
    l1_sparsity_penalty,
    mask_summary,
    soft_data_mask,
    sparsity_fraction,
)
from ltx_trainer.dyna_pruner.pipeline import (
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_ablation,
    table_main_results,
    table_robustness,
    table_sparsity_sensitivity,
)
from ltx_trainer.dyna_pruner.synergy import (
    aggregate_receptive_field,
    model_mask_from_importance,
    synchronized_masks,
)

__all__ = [
    "DynaPrunerConfig",
    "LIMITATIONS",
    "SparsityBudget",
    "aggregate_receptive_field",
    "apply_data_mask",
    "energy_aware_training_note",
    "evaluation_demo",
    "framework_card",
    "gopex_env_exports",
    "hard_mask_ste",
    "importance_from_temporal_variance",
    "l1_sparsity_penalty",
    "ltx_inference_plan",
    "ltx_prep_plan",
    "mask_summary",
    "model_mask_from_importance",
    "mse_task_loss",
    "pipeline_demo",
    "soft_data_mask",
    "sparsity_fraction",
    "synchronized_masks",
    "table_ablation",
    "table_main_results",
    "table_robustness",
    "table_sparsity_sensitivity",
    "total_loss",
]
