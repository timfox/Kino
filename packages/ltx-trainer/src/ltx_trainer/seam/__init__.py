"""SEAM — shortcut-aware scripted vs spontaneous detection (arXiv:2606.06837)."""

from ltx_trainer.seam.augmentation import augmentation_demo, inject_noise_bank
from ltx_trainer.seam.config import SeamConfig
from ltx_trainer.seam.eval import eval_smoke, pipeline_demo
from ltx_trainer.seam.fold import annotate_audio_save_data
from ltx_trainer.seam.mock import evaluation_smoke
from ltx_trainer.seam.model import classify_window, model_demo
from ltx_trainer.seam.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_full_training,
    table2_shortcut_ablation,
    table3_window_length,
    table4_adaptation_depth,
    table5_quantization,
    table6_backbone_screening,
)
from ltx_trainer.seam.preprocessing import preprocess_waveform, preprocessing_demo
from ltx_trainer.seam.sampling import sampling_demo, seam_aware_sample

__all__ = [
    "SeamConfig",
    "annotate_audio_save_data",
    "augmentation_demo",
    "benchmarks_bundle",
    "classify_window",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "inject_noise_bank",
    "model_demo",
    "pipeline_demo",
    "preprocess_waveform",
    "preprocessing_demo",
    "sampling_demo",
    "seam_aware_sample",
    "table1_full_training",
    "table2_shortcut_ablation",
    "table3_window_length",
    "table4_adaptation_depth",
    "table5_quantization",
    "table6_backbone_screening",
]
