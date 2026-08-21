"""SimInsert: training-free video object insertion (Chen et al., arXiv:2605.23245)."""

from ltx_trainer.siminsert.attention import (
    apply_value_guidance,
    regional_attention_clone,
    sparse_attention_fusion,
)
from ltx_trainer.siminsert.config import BASELINES, PAPER_METRICS, SimInsertConfig
from ltx_trainer.siminsert.latent import (
    flow_matching_forward,
    latent_refresh,
    reconstruction_latent,
)
from ltx_trainer.siminsert.metrics import evaluate_background_metrics
from ltx_trainer.siminsert.pipeline import (
    benchmark_table,
    dataset_card,
    demo_video_latents,
    evaluate_insertion,
    propagate_demo,
    siminsert_denoise_step,
)

__all__ = [
    "BASELINES",
    "PAPER_METRICS",
    "SimInsertConfig",
    "apply_value_guidance",
    "benchmark_table",
    "dataset_card",
    "demo_video_latents",
    "evaluate_background_metrics",
    "evaluate_insertion",
    "flow_matching_forward",
    "latent_refresh",
    "propagate_demo",
    "reconstruction_latent",
    "regional_attention_clone",
    "siminsert_denoise_step",
    "sparse_attention_fusion",
]
