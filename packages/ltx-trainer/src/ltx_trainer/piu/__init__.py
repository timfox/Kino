"""PIU: Proximity-guided Identity Unlearning (Bakir et al., arXiv:2605.22311)."""

from ltx_trainer.piu.anchor import (
    build_centroids_from_clusters,
    candidate_anchor_set,
    cosine_similarity,
    identity_centroid,
    select_anchor_identity,
)
from ltx_trainer.piu.conditioning import pad_arcface_to_clip, sample_forget_batch, synthetic_forget_condition
from ltx_trainer.piu.config import BASELINES, PIUConfig
from ltx_trainer.piu.layers import SURGICAL_BLOCKS, is_surgical_param
from ltx_trainer.piu.losses import forget_loss, forget_target_noise, preserve_loss, total_piu_loss
from ltx_trainer.piu.metrics import evaluate_unlearning, identity_score_matching, srk_score
from ltx_trainer.piu.pipeline import (
    DemoNoisePredictor,
    ablation_eta_table,
    ablation_lambda_table,
    ablation_layers_table,
    ablation_table_components,
    ablation_tau_table,
    benchmark_table_sota,
    dataset_card,
    demo_anchor_selection,
    demo_identity_clusters,
    training_step,
)

__all__ = [
    "BASELINES",
    "DemoNoisePredictor",
    "PIUConfig",
    "SURGICAL_BLOCKS",
    "ablation_eta_table",
    "ablation_lambda_table",
    "ablation_layers_table",
    "ablation_table_components",
    "ablation_tau_table",
    "benchmark_table_sota",
    "build_centroids_from_clusters",
    "candidate_anchor_set",
    "cosine_similarity",
    "dataset_card",
    "demo_anchor_selection",
    "demo_identity_clusters",
    "evaluate_unlearning",
    "forget_loss",
    "forget_target_noise",
    "identity_centroid",
    "identity_score_matching",
    "is_surgical_param",
    "pad_arcface_to_clip",
    "preserve_loss",
    "sample_forget_batch",
    "select_anchor_identity",
    "srk_score",
    "synthetic_forget_condition",
    "total_piu_loss",
    "training_step",
]
