"""MixFake: mixed-audio deepfake detection benchmark + multi-stream prompt tuning (arXiv:2605.23201)."""

from ltx_trainer.mixfake.config import MixFakeConfig, MixFakeDatasetStats
from ltx_trainer.mixfake.layout import LIMITATIONS
from ltx_trainer.mixfake.mixing import MixLabel, background_label, foreground_label
from ltx_trainer.mixfake.pipeline import (
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_ablation_prompts,
    table_dataset_overview,
    table_in_the_wild_eer,
    table_mixfake_subtasks_eer,
    table_snr_highlights,
)
from ltx_trainer.mixfake.prompts import layer_input_width, texture_prompt
from ltx_trainer.mixfake.signals import feature_flux, teager_kaiser_sequence

__all__ = [
    "MixFakeConfig",
    "MixFakeDatasetStats",
    "MixLabel",
    "LIMITATIONS",
    "background_label",
    "evaluation_demo",
    "feature_flux",
    "foreground_label",
    "framework_card",
    "layer_input_width",
    "pipeline_demo",
    "table_ablation_prompts",
    "table_dataset_overview",
    "table_in_the_wild_eer",
    "table_mixfake_subtasks_eer",
    "table_snr_highlights",
    "teager_kaiser_sequence",
    "texture_prompt",
]
