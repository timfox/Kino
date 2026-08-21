"""LoRA-Key: user-centric LoRA watermarking (arXiv:2605.29569)."""

from ltx_trainer.lora_key.config import LoRAKeyConfig
from ltx_trainer.lora_key.gop import (
    cosine_similarity_flat,
    gradient_orthogonal_projection,
    projection_coefficient,
)
from ltx_trainer.lora_key.losses import estimate_z0, semantic_cosine_loss, watermark_consistency_loss
from ltx_trainer.lora_key.lora_math import lora_delta, lora_forward, merge_loras, multi_lora_merge
from ltx_trainer.lora_key.ltx_bridge import LoRAKeyLTXBridge, ltx_integration_notes
from ltx_trainer.lora_key.metrics import (
    gop_cosine_ablation,
    table_i_fidelity_watermark,
    table_ii_robustness,
    table_iii_scalability,
    table_iv_cross_architecture,
    table_v_community_lora,
    table_vii_rank_ablation,
)
from ltx_trainer.lora_key.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    training_step_demo,
)
from ltx_trainer.lora_key.prior import LatentWatermarkPrior, prior_loss
from ltx_trainer.lora_key.verification import binomial_fpr, matching_bits, verification_threshold, verify_ownership

__all__ = [
    "LoRAKeyConfig",
    "LoRAKeyLTXBridge",
    "LatentWatermarkPrior",
    "benchmark_manifest",
    "binomial_fpr",
    "cosine_similarity_flat",
    "evaluation_demo",
    "estimate_z0",
    "framework_card",
    "gop_cosine_ablation",
    "gradient_orthogonal_projection",
    "lora_delta",
    "lora_forward",
    "ltx_integration_notes",
    "matching_bits",
    "merge_loras",
    "multi_lora_merge",
    "paper_limitations",
    "prior_loss",
    "projection_coefficient",
    "semantic_cosine_loss",
    "table_i_fidelity_watermark",
    "table_ii_robustness",
    "table_iii_scalability",
    "table_iv_cross_architecture",
    "table_v_community_lora",
    "table_vii_rank_ablation",
    "training_step_demo",
    "verification_threshold",
    "verify_ownership",
    "watermark_consistency_loss",
]
