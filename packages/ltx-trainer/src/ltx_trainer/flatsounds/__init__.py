"""FlatSounds: V2A physical benchmark (arXiv:2605.30339)."""

from ltx_trainer.flatsounds.config import FlatSoundsConfig
from ltx_trainer.flatsounds.hooks import (
    attach_flatsounds_to_preprocess_meta,
    caption_hint_lines,
    flatsounds_enabled,
    flatsounds_preprocess_extra,
    flatsounds_user_prompt_lines,
    ltx_video_prompt_suffix,
    merge_preprocess_extra,
    qa_manifest_directory,
)
from ltx_trainer.flatsounds.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)

__all__ = [
    "FlatSoundsConfig",
    "attach_flatsounds_to_preprocess_meta",
    "caption_hint_lines",
    "evaluation_demo",
    "evaluation_smoke",
    "flatsounds_enabled",
    "flatsounds_preprocess_extra",
    "flatsounds_user_prompt_lines",
    "framework_card",
    "knowledge_card",
    "ltx_video_prompt_suffix",
    "merge_preprocess_extra",
    "qa_manifest_directory",
]
