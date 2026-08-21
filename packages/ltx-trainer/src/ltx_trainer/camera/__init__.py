"""CAMERA — unsupervised TAGFD under semantic camouflage (arXiv:2605.20032)."""

from ltx_trainer.camera.config import CAMERAConfig
from ltx_trainer.camera.detector import fraud_scores
from ltx_trainer.camera.layout import LIMITATIONS
from ltx_trainer.camera.losses import expert_loss, gating_entropy_loss, oc_bce_loss, total_loss
from ltx_trainer.camera.moe import context_informed_gating, ego_decoupled_moe_layer
from ltx_trainer.camera.pipeline import (
    benchmarks_bundle,
    dataset_statistics,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_i_main_comparison,
    table_ii_text_encoder_ablation,
    table_iii_expert_ablation,
)

__all__ = [
    "CAMERAConfig",
    "LIMITATIONS",
    "benchmarks_bundle",
    "context_informed_gating",
    "dataset_statistics",
    "ego_decoupled_moe_layer",
    "evaluation_demo",
    "expert_loss",
    "framework_card",
    "fraud_scores",
    "gating_entropy_loss",
    "oc_bce_loss",
    "pipeline_demo",
    "table_i_main_comparison",
    "table_ii_text_encoder_ablation",
    "table_iii_expert_ablation",
    "total_loss",
]
