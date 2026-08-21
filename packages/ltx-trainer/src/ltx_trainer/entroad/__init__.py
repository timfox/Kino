"""EntroAD: structural entropy-guided ZSAD (arXiv:2605.28630)."""

from ltx_trainer.entroad.config import EntroADConfig
from ltx_trainer.entroad.dual_branch import fuse_branch_maps, image_score_from_map
from ltx_trainer.entroad.entropy import structural_entropy_map
from ltx_trainer.entroad.hooks import (
    attach_entroad_to_preprocess_meta,
    caption_hint_lines,
    entroad_enabled,
    entroad_preprocess_extra,
    entroad_user_prompt_lines,
    ltx_video_prompt_suffix,
    merge_preprocess_extra,
    qa_manifest_directory,
    score_anomaly_caption,
)
from ltx_trainer.entroad.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
from ltx_trainer.entroad.routing import routed_token_pair

__all__ = [
    "EntroADConfig",
    "attach_entroad_to_preprocess_meta",
    "caption_hint_lines",
    "entroad_enabled",
    "entroad_preprocess_extra",
    "entroad_user_prompt_lines",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "fuse_branch_maps",
    "image_score_from_map",
    "knowledge_card",
    "ltx_video_prompt_suffix",
    "merge_preprocess_extra",
    "qa_manifest_directory",
    "routed_token_pair",
    "score_anomaly_caption",
    "structural_entropy_map",
]
