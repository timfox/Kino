"""UAV-OVO: out-of-viewpoint UAV action recognition (Xia et al., arXiv:2605.25615)."""

from ltx_trainer.uav_ovo.config import UAVOVOConfig
from ltx_trainer.uav_ovo.later import (
    OnlineTargetCenter,
    build_lora_subspace,
    later_recenter,
    orthogonal_projector,
)
from ltx_trainer.uav_ovo.metrics import harmonic_mean, performance_drop
from ltx_trainer.uav_ovo.pipeline import (
    evaluation_demo,
    framework_card,
    table_alpha_ablation,
    table_lora_rank_ablation,
    table_main_results,
    table_projection_ablation,
    table_split_statistics,
    training_step_demo,
)
from ltx_trainer.uav_ovo.view_score import ViewSplit, assign_split, pitch_offset_deg, video_view_score

__all__ = [
    "OnlineTargetCenter",
    "UAVOVOConfig",
    "ViewSplit",
    "assign_split",
    "build_lora_subspace",
    "evaluation_demo",
    "framework_card",
    "harmonic_mean",
    "later_recenter",
    "orthogonal_projector",
    "performance_drop",
    "pitch_offset_deg",
    "table_alpha_ablation",
    "table_lora_rank_ablation",
    "table_main_results",
    "table_projection_ablation",
    "table_split_statistics",
    "training_step_demo",
    "video_view_score",
]
