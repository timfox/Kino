"""PESD-ViT: parameter-efficient subspace-decoupling ViT for NAFLD NAS (arXiv:2605.29852)."""

from ltx_trainer.pesd_vit.adapter import TaskAdapter, adapter_param_count
from ltx_trainer.pesd_vit.config import PesdVitConfig
from ltx_trainer.pesd_vit.hooks import (
    attach_pesd_vit_to_preprocess_meta,
    caption_hint_lines,
    ltx_video_prompt_suffix,
    merge_preprocess_extra,
    pesd_vit_enabled,
    pesd_vit_preprocess_extra,
    pesd_vit_user_prompt_lines,
    qa_manifest_directory,
)
from ltx_trainer.pesd_vit.mtl_loss import homoscedastic_mtl_loss, total_training_loss
from ltx_trainer.pesd_vit.ortho import orthogonal_decoupling_loss
from ltx_trainer.pesd_vit.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)

__all__ = [
    "PesdVitConfig",
    "TaskAdapter",
    "adapter_param_count",
    "attach_pesd_vit_to_preprocess_meta",
    "caption_hint_lines",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "homoscedastic_mtl_loss",
    "knowledge_card",
    "ltx_video_prompt_suffix",
    "merge_preprocess_extra",
    "orthogonal_decoupling_loss",
    "pesd_vit_enabled",
    "pesd_vit_preprocess_extra",
    "pesd_vit_user_prompt_lines",
    "qa_manifest_directory",
    "total_training_loss",
]
