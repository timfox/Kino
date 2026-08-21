"""Squeeze MLLM capacity for subject-driven generation (arXiv:2605.26111)."""

from ltx_trainer.squeeze_mllm.config import SqueezeMLLMConfig
from ltx_trainer.squeeze_mllm.denoising import StageMasks, mask_schedule, stage_masks
from ltx_trainer.squeeze_mllm.dla import DLAOutput, DualLayerAggregator, synthetic_mllm_layer_stack
from ltx_trainer.squeeze_mllm.lap import LayerwiseAttentionPooling
from ltx_trainer.squeeze_mllm.layout import architecture_layout, paper_limitations
from ltx_trainer.squeeze_mllm.pipeline import (
    dla_demo,
    evaluation_demo,
    framework_card,
    single_lap_vs_dla_tradeoff,
    table_copy_paste,
    table_denoising_sensitivity,
    table_dreambench,
    table_mllm_connection_ablation,
    table_reasoning_clip_t,
    table_two_stage_training,
)

__all__ = [
    "DLAOutput",
    "DualLayerAggregator",
    "LayerwiseAttentionPooling",
    "SqueezeMLLMConfig",
    "StageMasks",
    "architecture_layout",
    "dla_demo",
    "evaluation_demo",
    "framework_card",
    "mask_schedule",
    "paper_limitations",
    "single_lap_vs_dla_tradeoff",
    "stage_masks",
    "synthetic_mllm_layer_stack",
    "table_copy_paste",
    "table_denoising_sensitivity",
    "table_dreambench",
    "table_mllm_connection_ablation",
    "table_reasoning_clip_t",
    "table_two_stage_training",
]
