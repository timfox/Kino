"""DRFusion: drift-resilient IR–visible video fusion (Li et al., arXiv:2605.25775).

Reference: history guidance Eq. (6), VAE temporal loss, condition adapter, paper Tables 1–4.
Full 3D-DiT training on HDO/M3SVD/NOT-156/VTMOT is external — see https://github.com/xhhaoyan/DRFusion
"""

from ltx_trainer.drfusion.adapter import ConditionAdapter
from ltx_trainer.drfusion.config import DRFusionConfig
from ltx_trainer.drfusion.history import (
    HistoryMode,
    build_history_window,
    noise_modulate,
    stabilized_history_guidance,
)
from ltx_trainer.drfusion.losses import (
    cooperative_latent_update,
    fusion_pixel_loss,
    latent_refinement_energy,
    latent_temporal_loss,
)
from ltx_trainer.drfusion.model import DRFusionVelocityHead
from ltx_trainer.drfusion.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_not156,
    table_fusion_quality,
    table_object_tracking,
    table_temporal_stability,
    training_step_demo,
)

__all__ = [
    "ConditionAdapter",
    "DRFusionConfig",
    "DRFusionVelocityHead",
    "HistoryMode",
    "build_history_window",
    "cooperative_latent_update",
    "evaluation_demo",
    "framework_card",
    "fusion_pixel_loss",
    "latent_refinement_energy",
    "latent_temporal_loss",
    "noise_modulate",
    "stabilized_history_guidance",
    "table_ablation_not156",
    "table_fusion_quality",
    "table_object_tracking",
    "table_temporal_stability",
    "training_step_demo",
]
