"""LLaVA-OneVision-2 codec-aligned MLLM (Liang et al., arXiv:2605.25979)."""

from ltx_trainer.llava_ov2.config import LLaVAOV2Config
from ltx_trainer.llava_ov2.codec_stream import (
    adaptive_gop_boundaries,
    bit_cost_per_bin,
    block_scores_from_saliency,
    motion_residual_saliency,
)
from ltx_trainer.llava_ov2.grouper import (
    codec_gop_groups,
    fixed_four_slot_groups,
    group_visible_mask,
    image_single_group,
)
from ltx_trainer.llava_ov2.pipeline import (
    evaluation_demo,
    framework_card,
    table_codec_frame_budget_sweep,
    table_codec_vs_uniform,
    table_spatial_benchmarks,
    table_tracking_jf,
    table_training_stages,
    table_video_benchmarks,
    training_step_demo,
)

__all__ = [
    "LLaVAOV2Config",
    "adaptive_gop_boundaries",
    "bit_cost_per_bin",
    "block_scores_from_saliency",
    "codec_gop_groups",
    "evaluation_demo",
    "fixed_four_slot_groups",
    "framework_card",
    "group_visible_mask",
    "image_single_group",
    "motion_residual_saliency",
    "table_codec_frame_budget_sweep",
    "table_codec_vs_uniform",
    "table_spatial_benchmarks",
    "table_tracking_jf",
    "table_training_stages",
    "table_video_benchmarks",
    "training_step_demo",
]
