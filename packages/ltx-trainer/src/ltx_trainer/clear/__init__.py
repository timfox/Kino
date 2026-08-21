"""CLEAR concept-layer erasure for T2V diffusion (Xie et al., arXiv:2605.25941)."""

from ltx_trainer.clear.config import CLEARConfig
from ltx_trainer.clear.gumbel import gumbel_softmax, select_layer_index
from ltx_trainer.clear.losses import lcon_separability, lsae_reconstruction
from ltx_trainer.clear.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_layer_search,
    table_cogvideo_objects,
    table_gamma_ablation,
    table_nudity_wan,
    table_wan_objects,
    training_step_demo,
)
from ltx_trainer.clear.sae import (
    SparseAutoencoder,
    concept_erasure,
    shared_mask,
    specificity_mask,
)

__all__ = [
    "CLEARConfig",
    "SparseAutoencoder",
    "concept_erasure",
    "evaluation_demo",
    "framework_card",
    "gumbel_softmax",
    "lcon_separability",
    "lsae_reconstruction",
    "select_layer_index",
    "shared_mask",
    "specificity_mask",
    "table_ablation_layer_search",
    "table_cogvideo_objects",
    "table_gamma_ablation",
    "table_nudity_wan",
    "table_wan_objects",
    "training_step_demo",
]
