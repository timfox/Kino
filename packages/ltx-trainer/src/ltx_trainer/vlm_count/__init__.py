"""VLM visual counting bottleneck stub (arXiv:2605.30170)."""

from ltx_trainer.vlm_count.config import VlmCountConfig
from ltx_trainer.vlm_count.hooks import (
    attach_vlm_count_to_preprocess_meta,
    merge_preprocess_extra,
    qa_manifest_directory,
    vlm_count_enabled,
    vlm_count_preprocess_extra,
    vlm_count_user_prompt_lines,
)
from ltx_trainer.vlm_count.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)

__all__ = [
    "VlmCountConfig",
    "attach_vlm_count_to_preprocess_meta",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "merge_preprocess_extra",
    "qa_manifest_directory",
    "vlm_count_enabled",
    "vlm_count_preprocess_extra",
    "vlm_count_user_prompt_lines",
]
