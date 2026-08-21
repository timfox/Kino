"""ControlLight: controllable low-light enhancement (Yang et al., arXiv:2605.25569).

Light100K Retinex interpolation, misalignment-aware weighted flow matching (L_wFM),
strength-scaled LoRA, benchmarks, and evaluation. FLUX.2-klein fine-tuning is external.
"""

from ltx_trainer.controllight.benchmarks import benchmarks_bundle
from ltx_trainer.controllight.config import ControlLightConfig, Light100KGroup
from ltx_trainer.controllight.edges import (
    edge_difference_map,
    misalignment_weight_map,
    structural_edge_response,
)
from ltx_trainer.controllight.evaluation import (
    enhance_trajectory,
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
from ltx_trainer.controllight.light100k import (
    build_light100k_from_paths,
    compare_interpolation_strategies,
    edge_consistency_score,
    make_training_group,
    pair_passes_edge_filter,
)
from ltx_trainer.controllight.li_lpips import li_lpips_proxy, lwfm_ablation_smoke
from ltx_trainer.controllight.lora import StrengthScaledLoRA
from ltx_trainer.controllight.losses import (
    flow_matching_loss,
    interpolate_latent,
    velocity_target,
    weighted_flow_matching_loss,
)
from ltx_trainer.controllight.metrics import clip_direction_score, delta_smooth, trajectory_lpips_distances
from ltx_trainer.controllight.nr_metrics import interpolation_trajectory_metrics, musiq_proxy, niqe_proxy
from ltx_trainer.controllight.pipeline import controllight_training_loss, prepare_training_batch, select_pseudo_target
from ltx_trainer.controllight.retinex import (
    alpha_blend_interpolate,
    build_light100k_group,
    retinex_interpolate,
)
from ltx_trainer.controllight.train_step import build_training_batch, training_step, training_step_demo

__all__ = [
    "ControlLightConfig",
    "Light100KGroup",
    "StrengthScaledLoRA",
    "alpha_blend_interpolate",
    "benchmarks_bundle",
    "build_light100k_from_paths",
    "build_light100k_group",
    "build_training_batch",
    "clip_direction_score",
    "compare_interpolation_strategies",
    "controllight_training_loss",
    "delta_smooth",
    "edge_consistency_score",
    "edge_difference_map",
    "enhance_trajectory",
    "evaluation_demo",
    "evaluation_smoke",
    "flow_matching_loss",
    "framework_card",
    "interpolation_trajectory_metrics",
    "interpolate_latent",
    "knowledge_card",
    "li_lpips_proxy",
    "lwfm_ablation_smoke",
    "make_training_group",
    "misalignment_weight_map",
    "musiq_proxy",
    "niqe_proxy",
    "pair_passes_edge_filter",
    "prepare_training_batch",
    "retinex_interpolate",
    "select_pseudo_target",
    "structural_edge_response",
    "training_step",
    "training_step_demo",
    "trajectory_lpips_distances",
    "velocity_target",
    "weighted_flow_matching_loss",
]
