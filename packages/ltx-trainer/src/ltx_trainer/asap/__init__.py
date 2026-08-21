"""ASAP: Alignment via Synthetic Anatomical Preference (Li et al., arXiv:2605.25759).

Reference utilities for HAP preference synthesis gates, localized margin-bounded DPO,
and HAF-Bench metrics. FLUX/SDXL LoRA training and external tools (DWPose, GroundingSAM2,
inpainting) are **not** bundled — pass tensors and masks in.
"""

from ltx_trainer.asap.config import ASAPConfig, HAFBenchConfig
from ltx_trainer.asap.degradation import (
    binary_anatomy_mask,
    corrupt_skeleton_connections,
    image_space_degraded_negative,
    region_mse,
)
from ltx_trainer.asap.haf_bench import (
    HAF_BENCH_SAMPLE_PROMPTS,
    HAFCategory,
    anatomical_error_rate,
    category_aer_report,
    relative_superiority_index,
    vlm_judge_stub,
)
from ltx_trainer.asap.hap import HAPPair, hap_batch_collate, pair_passes_hap_filters
from ltx_trainer.asap.losses import (
    anatomical_alignment_loss,
    diffusion_dpo_loss,
    flow_matching_loss,
    localized_velocity_mse,
    margin_bounded_loss,
    preference_gap,
)
from ltx_trainer.asap.pipeline import asap_training_loss, fm_losses_for_pair, synthetic_training_step

__all__ = [
    "ASAPConfig",
    "HAFBenchConfig",
    "HAPPair",
    "HAF_BENCH_SAMPLE_PROMPTS",
    "HAFCategory",
    "anatomical_alignment_loss",
    "anatomical_error_rate",
    "asap_training_loss",
    "binary_anatomy_mask",
    "category_aer_report",
    "corrupt_skeleton_connections",
    "diffusion_dpo_loss",
    "flow_matching_loss",
    "fm_losses_for_pair",
    "hap_batch_collate",
    "image_space_degraded_negative",
    "localized_velocity_mse",
    "margin_bounded_loss",
    "pair_passes_hap_filters",
    "preference_gap",
    "region_mse",
    "relative_superiority_index",
    "synthetic_training_step",
    "vlm_judge_stub",
]
