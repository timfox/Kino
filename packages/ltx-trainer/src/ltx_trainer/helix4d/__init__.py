"""Helix4D complex 4D mesh generation (Yenphraphai et al., arXiv:2605.26109)."""

from ltx_trainer.helix4d.attention import (
    apply_cross_frame_attention,
    frame_token_offsets,
    mask_pattern_name,
    masked_attention_logits,
    sliding_window_anchor_mask,
)
from ltx_trainer.helix4d.conditioning import (
    build_frame_timesteps,
    flow_matching_loss_mask,
    inject_anchor_latent,
    trellis2_stage_order,
)
from ltx_trainer.helix4d.config import Helix4DConfig
from ltx_trainer.helix4d.pipeline import (
    evaluation_demo,
    framework_card,
    table_actionbench_texverse,
    table_attention_patterns,
    table_component_ablation,
    table_helix4d_bench,
    table_rope_ratio_ablation,
)
from ltx_trainer.helix4d.rope import (
    apply_rope,
    rope_4d,
    rope_frequencies,
    spatial_rope_3d,
    split_spatial_rope,
    temporal_rope_1d,
)

__all__ = [
    "Helix4DConfig",
    "apply_cross_frame_attention",
    "apply_rope",
    "build_frame_timesteps",
    "evaluation_demo",
    "flow_matching_loss_mask",
    "framework_card",
    "frame_token_offsets",
    "inject_anchor_latent",
    "mask_pattern_name",
    "masked_attention_logits",
    "rope_4d",
    "rope_frequencies",
    "sliding_window_anchor_mask",
    "spatial_rope_3d",
    "split_spatial_rope",
    "table_actionbench_texverse",
    "table_attention_patterns",
    "table_component_ablation",
    "table_helix4d_bench",
    "table_rope_ratio_ablation",
    "temporal_rope_1d",
    "trellis2_stage_order",
]
