"""Subspace TBD — passive multi-target track-before-detect."""

from ltx_trainer.stbd.config import StbdConfig
from ltx_trainer.stbd.filter import (
    boundary_factor,
    mmse_position,
    nearly_constant_velocity_step,
    position_rmse,
)
from ltx_trainer.stbd.layout import LIMITATIONS
from ltx_trainer.stbd.likelihood import (
    aggregate_bingham_log_likelihood,
    aggregate_deterministic_log_likelihood,
    bingham_log_likelihood,
    deterministic_contribution_log_likelihood,
    snr_to_noise_variance,
)
from ltx_trainer.stbd.mock import compare_likelihoods_at_true_state, ring_microphones, simulate_normalized_mixture
from ltx_trainer.stbd.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_median_rmse,
    trajectory_example_rmse,
)
from ltx_trainer.stbd.steering import (
    mixing_matrix,
    normalize_observation,
    steering_vector_planar,
    subspace_projector,
)

__all__ = [
    "LIMITATIONS",
    "StbdConfig",
    "aggregate_bingham_log_likelihood",
    "aggregate_deterministic_log_likelihood",
    "benchmarks_bundle",
    "bingham_log_likelihood",
    "boundary_factor",
    "compare_likelihoods_at_true_state",
    "deterministic_contribution_log_likelihood",
    "evaluation_demo",
    "framework_card",
    "headline_results",
    "mixing_matrix",
    "mmse_position",
    "nearly_constant_velocity_step",
    "normalize_observation",
    "pipeline_demo",
    "position_rmse",
    "ring_microphones",
    "simulate_normalized_mixture",
    "snr_to_noise_variance",
    "steering_vector_planar",
    "subspace_projector",
    "table_i_median_rmse",
    "trajectory_example_rmse",
]
