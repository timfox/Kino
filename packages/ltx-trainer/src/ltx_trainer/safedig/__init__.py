"""SafeDIG — robust DiT safety steering (arXiv:2605.30049).

Position-aware sparse autoencoder steering with robustness routing and
decoder-only domain transfer for FLUX / SD3.5-class DiTs.
"""

from ltx_trainer.safedig.config import InterventionHook, SafeDIGConfig
from ltx_trainer.safedig.metrics import (
    fuse_unsafe_flags,
    line_level_asr,
    prompt_level_asr,
    table1_line_level,
    table1_main_results,
    table2_transfer_positions,
    table6_ablation_snippet,
)
from ltx_trainer.safedig.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    training_step_demo,
)
from ltx_trainer.safedig.routing import RoutedPair, rank_interventions, robustness_score
from ltx_trainer.safedig.sae import (
    SparseAutoencoder,
    contrast_activation,
    decoder_only_transfer,
    default_hooks,
    train_source_sae,
)
from ltx_trainer.safedig.steering import (
    blend_steering,
    estimate_harmful_mask,
    repel_steering,
    steer_activation,
)

__all__ = [
    "InterventionHook",
    "RoutedPair",
    "SafeDIGConfig",
    "SparseAutoencoder",
    "benchmark_manifest",
    "blend_steering",
    "contrast_activation",
    "decoder_only_transfer",
    "default_hooks",
    "estimate_harmful_mask",
    "evaluation_demo",
    "framework_card",
    "fuse_unsafe_flags",
    "line_level_asr",
    "paper_limitations",
    "prompt_level_asr",
    "rank_interventions",
    "repel_steering",
    "robustness_score",
    "steer_activation",
    "table1_line_level",
    "table1_main_results",
    "table2_transfer_positions",
    "table6_ablation_snippet",
    "train_source_sae",
    "training_step_demo",
]
