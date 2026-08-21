"""Plug-in evidential deep learning stub (arXiv:2605.22746)."""

from ltx_trainer.plug_edl.config import PlugEdlConfig
from ltx_trainer.plug_edl.evidential import (
    dirichlet_params,
    exp_evidence,
    project_dirichlet,
    simplified_classifier,
    softmax_from_logits,
    softplus_evidence,
)
from ltx_trainer.plug_edl.layout import LIMITATIONS
from ltx_trainer.plug_edl.losses import (
    approximation_remainder_bound,
    classical_edl_mse_toy,
    edl_mse_variance_term,
    plug_in_cross_entropy,
    plug_in_mse,
)
from ltx_trainer.plug_edl.metrics import (
    normalized_entropy,
    selective_prediction_metrics,
    vacuity,
)
from ltx_trainer.plug_edl.mock import evaluation_smoke
from ltx_trainer.plug_edl.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_model_variants,
    table2_selective_prediction_entropy,
    table2_selective_prediction_vacuity_excerpt,
)

__all__ = [
    "LIMITATIONS",
    "PlugEdlConfig",
    "approximation_remainder_bound",
    "benchmarks_bundle",
    "classical_edl_mse_toy",
    "dirichlet_params",
    "edl_mse_variance_term",
    "evaluation_demo",
    "evaluation_smoke",
    "exp_evidence",
    "framework_card",
    "headline_results",
    "normalized_entropy",
    "plug_in_cross_entropy",
    "plug_in_mse",
    "project_dirichlet",
    "selective_prediction_metrics",
    "simplified_classifier",
    "softmax_from_logits",
    "softplus_evidence",
    "table1_model_variants",
    "table2_selective_prediction_entropy",
    "table2_selective_prediction_vacuity_excerpt",
    "vacuity",
]
