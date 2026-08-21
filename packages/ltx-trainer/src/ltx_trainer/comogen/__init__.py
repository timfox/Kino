"""CoMoGen: mask-guided controllable video generation (Meric et al., arXiv:2605.22996)."""

from ltx_trainer.comogen.attention import (
    attention_score,
    demo_attention_maps,
    identify_motion_layers,
    layer_attention_scores,
)
from ltx_trainer.comogen.config import BASELINES, DATASETS, CoMoGenConfig
from ltx_trainer.comogen.mask_adapter import MaskAdapter, downsample_mask_sequence, mask_to_delta
from ltx_trainer.comogen.losses import flow_matching_loss, flow_matching_step_targets
from ltx_trainer.comogen.metrics import evaluate_video, tracking_jf_hota
from ltx_trainer.comogen.pipeline import (
    MotionLayerLoRA,
    ablation_table_behave,
    benchmark_table_behave,
    benchmark_table_clevrer,
    comogen_denoise_step,
    dataset_card,
    demo_mask_sequence,
    evaluate_comogen,
    layer_skip_benchmark,
    motion_layer_analysis,
    propagate_demo,
    sparsify_mask_sequence,
)
from ltx_trainer.comogen.schedule import cosine_schedule, cosine_step_weight, inject_latent_residual

__all__ = [
    "BASELINES",
    "CoMoGenConfig",
    "DATASETS",
    "MaskAdapter",
    "MotionLayerLoRA",
    "attention_score",
    "ablation_table_behave",
    "benchmark_table_behave",
    "benchmark_table_clevrer",
    "comogen_denoise_step",
    "cosine_schedule",
    "cosine_step_weight",
    "dataset_card",
    "demo_attention_maps",
    "demo_mask_sequence",
    "downsample_mask_sequence",
    "evaluate_comogen",
    "evaluate_video",
    "flow_matching_loss",
    "flow_matching_step_targets",
    "layer_skip_benchmark",
    "identify_motion_layers",
    "inject_latent_residual",
    "layer_attention_scores",
    "mask_to_delta",
    "motion_layer_analysis",
    "propagate_demo",
    "sparsify_mask_sequence",
    "tracking_jf_hota",
]
