"""MTLLFM: multimodal-temporal laughter localization (Hanania et al., arXiv:2605.25409)."""

from ltx_trainer.mtllfm.config import MTLLFMConfig
from ltx_trainer.mtllfm.gating import adaptive_modality_gating
from ltx_trainer.mtllfm.localization import localize_laughter, sharpen_attention
from ltx_trainer.mtllfm.pooling import temporal_softmax_pool
from ltx_trainer.mtllfm.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_sportspress,
    table_dataset_stats,
    table_downstream_reasoning,
    table_foundation_comparison,
    training_step_demo,
)

__all__ = [
    "MTLLFMConfig",
    "adaptive_modality_gating",
    "evaluation_demo",
    "framework_card",
    "localize_laughter",
    "sharpen_attention",
    "table_ablation_sportspress",
    "table_dataset_stats",
    "table_downstream_reasoning",
    "table_foundation_comparison",
    "temporal_softmax_pool",
    "training_step_demo",
]
