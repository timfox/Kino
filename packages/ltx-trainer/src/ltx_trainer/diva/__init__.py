"""DIVA: harnessing representation divergence in unified multimodal models (Lu et al., arXiv:2605.25328)."""

from ltx_trainer.diva.config import DIVAConfig
from ltx_trainer.diva.factorization import (
    DualFactorization,
    GatedFactorEncoder,
    inject_logits,
    orthogonality_loss,
    pool_image_tokens,
)
from ltx_trainer.diva.geometry import effective_rank, effective_rank_increment, reconstruction_residual
from ltx_trainer.diva.layout import LIMITATIONS
from ltx_trainer.diva.mock import AnchorSample, make_toy_flows, random_mask_ratio
from ltx_trainer.diva.mutual_info import (
    directed_shared_alignment,
    info_nce_shared,
    nce_club_unique_upper_bound,
    stage2_total_loss,
)
from ltx_trainer.diva.pipeline import (
    evaluation_demo,
    framework_card,
    stage1_demo,
    stage2_demo,
    table_ablation_show_o,
    table_lambda_uni_sensitivity,
    table_main_results,
    table_post_training_comparison,
    training_step_demo,
)

__all__ = [
    "AnchorSample",
    "DIVAConfig",
    "DualFactorization",
    "GatedFactorEncoder",
    "LIMITATIONS",
    "directed_shared_alignment",
    "effective_rank",
    "effective_rank_increment",
    "evaluation_demo",
    "framework_card",
    "info_nce_shared",
    "inject_logits",
    "make_toy_flows",
    "nce_club_unique_upper_bound",
    "orthogonality_loss",
    "pool_image_tokens",
    "random_mask_ratio",
    "reconstruction_residual",
    "stage1_demo",
    "stage2_demo",
    "stage2_total_loss",
    "table_ablation_show_o",
    "table_lambda_uni_sensitivity",
    "table_main_results",
    "table_post_training_comparison",
    "training_step_demo",
]
