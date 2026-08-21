"""Causal spatio-temporal sound field reconstruction (arXiv:2605.20403)."""

from ltx_trainer.causal_st_sfr.config import CausalStSfrConfig, DtuMeasuredCard, SimGeometryCard
from ltx_trainer.causal_st_sfr.kernel import (
    diffuse_far_field_normalized_coherence,
    discrete_covariance_C,
    kappa_band_limited,
)
from ltx_trainer.causal_st_sfr.layout import LIMITATIONS, circular_mic_array_xy, fibonacci_sphere_points
from ltx_trainer.causal_st_sfr.lmmse import (
    build_covariance_blocks,
    lmmse_posterior_mean,
    trace_posterior_variance_masked,
)
from ltx_trainer.causal_st_sfr.mock import evaluation_smoke
from ltx_trainer.causal_st_sfr.pipeline import (
    baselines_catalog,
    benchmarks_bundle,
    evaluation_demo,
    experiment_setup,
    figure_excerpts,
    framework_card,
)

__all__ = [
    "LIMITATIONS",
    "CausalStSfrConfig",
    "DtuMeasuredCard",
    "SimGeometryCard",
    "baselines_catalog",
    "benchmarks_bundle",
    "build_covariance_blocks",
    "circular_mic_array_xy",
    "diffuse_far_field_normalized_coherence",
    "discrete_covariance_C",
    "evaluation_demo",
    "evaluation_smoke",
    "experiment_setup",
    "figure_excerpts",
    "fibonacci_sphere_points",
    "framework_card",
    "kappa_band_limited",
    "lmmse_posterior_mean",
    "trace_posterior_variance_masked",
]
