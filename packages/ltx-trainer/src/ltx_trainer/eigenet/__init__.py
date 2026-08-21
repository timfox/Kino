"""EIGENET: geometry-informed few-shot novel view RIR prediction (arXiv:2605.28101)."""

from ltx_trainer.eigenet.config import EigeNetConfig
from ltx_trainer.eigenet.cvat import alternate_attention_block, cvat_forward
from ltx_trainer.eigenet.hooks import (
    attach_eigenet_to_preprocess_meta,
    caption_hint_lines,
    eigenet_enabled,
    eigenet_preprocess_extra,
    eigenet_user_prompt_lines,
    ltx_av_prompt_suffix,
    merge_preprocess_extra,
    qa_manifest_directory,
    score_spatial_audio_caption,
)
from ltx_trainer.eigenet.metrics import metric_triplet, sabine_t60_proxy
from ltx_trainer.eigenet.modulation import geometry_informed_modulate, spectrum_loss
from ltx_trainer.eigenet.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)

__all__ = [
    "EigeNetConfig",
    "alternate_attention_block",
    "attach_eigenet_to_preprocess_meta",
    "caption_hint_lines",
    "cvat_forward",
    "eigenet_enabled",
    "eigenet_preprocess_extra",
    "eigenet_user_prompt_lines",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "geometry_informed_modulate",
    "knowledge_card",
    "ltx_av_prompt_suffix",
    "merge_preprocess_extra",
    "metric_triplet",
    "qa_manifest_directory",
    "sabine_t60_proxy",
    "score_spatial_audio_caption",
    "spectrum_loss",
]
