"""PlanAudio: unified speech+sound from free-form text (arXiv:2605.28063)."""

from ltx_trainer.planaudio.config import PlanAudioConfig
from ltx_trainer.planaudio.cot import (
    downsample_af3_embeddings,
    format_sequence,
    latent_supervision_loss,
    total_loss,
)
from ltx_trainer.planaudio.hooks import (
    attach_planaudio_to_preprocess_meta,
    caption_hint_lines,
    ltx_av_prompt_suffix,
    merge_preprocess_extra,
    planaudio_enabled,
    planaudio_preprocess_extra,
    planaudio_user_prompt_lines,
    qa_manifest_directory,
    score_prompt_composition,
)
from ltx_trainer.planaudio.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
from ltx_trainer.planaudio.scenarios import classify_free_form_prompt, scenario_card
from ltx_trainer.planaudio.scoring import normalized_scenario_score, semantic_coverage_factor

__all__ = [
    "PlanAudioConfig",
    "attach_planaudio_to_preprocess_meta",
    "caption_hint_lines",
    "classify_free_form_prompt",
    "downsample_af3_embeddings",
    "evaluation_demo",
    "evaluation_smoke",
    "format_sequence",
    "framework_card",
    "knowledge_card",
    "latent_supervision_loss",
    "ltx_av_prompt_suffix",
    "merge_preprocess_extra",
    "normalized_scenario_score",
    "planaudio_enabled",
    "planaudio_preprocess_extra",
    "planaudio_user_prompt_lines",
    "qa_manifest_directory",
    "scenario_card",
    "score_prompt_composition",
    "semantic_coverage_factor",
    "total_loss",
]
