"""cSTMM — complex spherical Student's t mixture model for mask-based BSS."""

from ltx_trainer.cstmm.config import CstmmConfig
from ltx_trainer.cstmm.density import (
    limiting_case_label,
    log_density_unnormalized,
    watson_log_density_unnormalized,
)
from ltx_trainer.cstmm.geometry import is_valid_precision, normalize_observation, quadratic_form
from ltx_trainer.cstmm.layout import LIMITATIONS
from ltx_trainer.cstmm.mm import (
    build_canonical_a,
    hca_concentration,
    hca_eigenvalues,
    phi_auxiliary,
    responsibilities,
    scatter_matrix,
    update_weights,
)
from ltx_trainer.cstmm.mock import component_log_likelihood, random_unit_vectors, toy_mm_step
from ltx_trainer.cstmm.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    figure_i_model_recovery,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_sdri,
)

__all__ = [
    "CstmmConfig",
    "LIMITATIONS",
    "benchmarks_bundle",
    "build_canonical_a",
    "component_log_likelihood",
    "evaluation_demo",
    "figure_i_model_recovery",
    "framework_card",
    "hca_concentration",
    "hca_eigenvalues",
    "headline_results",
    "is_valid_precision",
    "limiting_case_label",
    "log_density_unnormalized",
    "normalize_observation",
    "phi_auxiliary",
    "pipeline_demo",
    "quadratic_form",
    "random_unit_vectors",
    "responsibilities",
    "scatter_matrix",
    "table_i_sdri",
    "toy_mm_step",
    "update_weights",
    "watson_log_density_unnormalized",
]
