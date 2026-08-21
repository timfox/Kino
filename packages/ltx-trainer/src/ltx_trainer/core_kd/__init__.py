"""CoRe-KD: complete-view reference-guided distillation for conversational MER (arXiv:2605.29590)."""

from ltx_trainer.core_kd.config import CoreKDConfig
from ltx_trainer.core_kd.hooks import (
    attach_core_kd_to_preprocess_meta,
    caption_hint_lines,
    core_kd_enabled,
    core_kd_preprocess_extra,
    core_kd_user_prompt_lines,
    ltx_video_prompt_suffix,
    merge_preprocess_extra,
    qa_manifest_directory,
)
from ltx_trainer.core_kd.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)

__all__ = [
    "CoreKDConfig",
    "attach_core_kd_to_preprocess_meta",
    "caption_hint_lines",
    "core_kd_enabled",
    "core_kd_preprocess_extra",
    "core_kd_user_prompt_lines",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "ltx_video_prompt_suffix",
    "merge_preprocess_extra",
    "qa_manifest_directory",
]
