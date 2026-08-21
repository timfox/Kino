"""SegCompass — SAE interpretable alignment for reasoning segmentation (arXiv:2605.22658)."""

from ltx_trainer.segcompass.activation import avg_active_features, instance_coverage_topk
from ltx_trainer.segcompass.alignment import AlignmentParadigm, paradigm_properties
from ltx_trainer.segcompass.codebook import aggregate_concept_slots, decode_sparse_pairs
from ltx_trainer.segcompass.config import MLLMBackbone, SAE_LAYERS, SegCompassConfig
from ltx_trainer.segcompass.grpo import clipped_policy_ratio, group_advantages
from ltx_trainer.segcompass.layout import LIMITATIONS
from ltx_trainer.segcompass.objectives import total_objective
from ltx_trainer.segcompass.decoder import decode_masks
from ltx_trainer.segcompass.encoder import encode_slot_concepts
from ltx_trainer.segcompass.heatmap import multi_slot_heatmaps
from ltx_trainer.segcompass.matching import hungarian_mean_overlap
from ltx_trainer.segcompass.metrics import cumulative_iou, generalized_iou
from ltx_trainer.segcompass.policy import mcot_rollout_spec, policy_logprob_ratio
from ltx_trainer.segcompass.pipeline import (
    evaluation_demo,
    framework_card,
    interpretability_analysis,
    interpretability_correlations,
    pipeline_demo,
    table_grpo_group_size,
    table_grefcoco,
    table_reasonseg_zero_shot,
    table_refcoco_full,
    table_refcoco_series,
    table_reward_ablation,
    table_training_mode_ablation,
    table_vision_backbone_ablation,
    training_curriculum,
    training_stack,
)
from ltx_trainer.segcompass.rewards import combined_reward, format_score, multi_object_mask_reward, soft_iou
from ltx_trainer.segcompass.sae import sae_reconstruction_loss, support_indices

__all__ = [
    "LIMITATIONS",
    "AlignmentParadigm",
    "MLLMBackbone",
    "SAE_LAYERS",
    "SegCompassConfig",
    "aggregate_concept_slots",
    "avg_active_features",
    "clipped_policy_ratio",
    "combined_reward",
    "cumulative_iou",
    "decode_masks",
    "decode_sparse_pairs",
    "encode_slot_concepts",
    "evaluation_demo",
    "format_score",
    "framework_card",
    "generalized_iou",
    "group_advantages",
    "hungarian_mean_overlap",
    "instance_coverage_topk",
    "interpretability_analysis",
    "interpretability_correlations",
    "mcot_rollout_spec",
    "multi_object_mask_reward",
    "multi_slot_heatmaps",
    "paradigm_properties",
    "pipeline_demo",
    "policy_logprob_ratio",
    "sae_reconstruction_loss",
    "soft_iou",
    "support_indices",
    "table_grpo_group_size",
    "table_grefcoco",
    "table_reasonseg_zero_shot",
    "table_refcoco_full",
    "table_refcoco_series",
    "table_reward_ablation",
    "table_training_mode_ablation",
    "table_vision_backbone_ablation",
    "total_objective",
    "training_curriculum",
    "training_stack",
]
