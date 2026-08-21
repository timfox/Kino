"""LaMo: self-supervised latent motion priors (Jiang et al., arXiv:2605.23878)."""

from ltx_trainer.lamo.config import LaMoConfig
from ltx_trainer.lamo.guidance import (
    apply_motion_prior_guidance,
    guidance_active,
    guidance_loss,
)
from ltx_trainer.lamo.heatmaps import motion_drift_heatmap, motion_field_heatmap
from ltx_trainer.lamo.latent_motion import (
    empirical_macro_drift,
    latent_delta,
    macro_drift,
    macro_drift_vector,
    select_strongest_motion_index,
)
from ltx_trainer.lamo.losses import (
    motion_drift_loss,
    predictor_loss,
    scale_normalized_drift_loss,
    schedule_weight,
    training_objective,
)
from ltx_trainer.lamo.pipeline import (
    build_predictor,
    evaluate_motion_readouts,
    lamo_training_step,
    train_predictor_step,
)
from ltx_trainer.lamo.predictor import MotionFieldPredictor

__all__ = [
    "LaMoConfig",
    "MotionFieldPredictor",
    "apply_motion_prior_guidance",
    "build_predictor",
    "empirical_macro_drift",
    "evaluate_motion_readouts",
    "guidance_active",
    "guidance_loss",
    "lamo_training_step",
    "latent_delta",
    "macro_drift",
    "macro_drift_vector",
    "motion_drift_heatmap",
    "motion_drift_loss",
    "motion_field_heatmap",
    "predictor_loss",
    "scale_normalized_drift_loss",
    "schedule_weight",
    "select_strongest_motion_index",
    "train_predictor_step",
    "training_objective",
]
