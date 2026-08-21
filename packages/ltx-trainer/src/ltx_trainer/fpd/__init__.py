"""Fixed-Point Distillation for discrete diffusion (Wang & Tong, arXiv:2605.21484)."""

from ltx_trainer.fpd.config import DINOV3_BLOCKS, DRIFT_BANDWIDTHS, FPDConfig
from ltx_trainer.fpd.drift import drift_loss, drift_vector
from ltx_trainer.fpd.masking import forward_mask_tokens, mask_probability, remask_fraction
from ltx_trainer.fpd.pipeline import (
    framework_card,
    table_ablation_drift_space,
    table_ablation_refinement_source,
    table_ablation_routing,
    table_geneval_text2image,
    table_imagenet_class_cond,
    training_step_demo,
)
from ltx_trainer.fpd.ste import (
    sample_hard_indices,
    soft_codebook_embedding,
    straight_through_embedding,
)

__all__ = [
    "DINOV3_BLOCKS",
    "DRIFT_BANDWIDTHS",
    "FPDConfig",
    "drift_loss",
    "drift_vector",
    "forward_mask_tokens",
    "framework_card",
    "mask_probability",
    "remask_fraction",
    "sample_hard_indices",
    "soft_codebook_embedding",
    "straight_through_embedding",
    "table_ablation_drift_space",
    "table_ablation_refinement_source",
    "table_ablation_routing",
    "table_geneval_text2image",
    "table_imagenet_class_cond",
    "training_step_demo",
]
