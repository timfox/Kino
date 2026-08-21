"""Sparse 4D bootstrapped cross-validation (arXiv:2605.19160)."""

from ltx_trainer.sparse4d_bcv.config import Sparse4dBcvConfig
from ltx_trainer.sparse4d_bcv.layout import COMPARISON_MODES, LIMITATIONS, RECOMMENDED_PRACTICE
from ltx_trainer.sparse4d_bcv.bootstrap import bootstrap_cv_summary, interlaced_cv_metric
from ltx_trainer.sparse4d_bcv.mock import bootstrap_demo, evaluation_smoke, nyquist_demo
from ltx_trainer.sparse4d_bcv.nyquist import (
    evenly_spaced_projection_angles,
    interlaced_time_indices,
    nyquist_velocity,
)
from ltx_trainer.sparse4d_bcv.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.sparse4d_bcv.tables import table1_metrics_summary, table_s1_ultrasparse_angles

__all__ = [
    "COMPARISON_MODES",
    "LIMITATIONS",
    "RECOMMENDED_PRACTICE",
    "Sparse4dBcvConfig",
    "benchmarks_bundle",
    "bootstrap_cv_summary",
    "bootstrap_demo",
    "interlaced_cv_metric",
    "evaluation_demo",
    "evaluation_smoke",
    "evenly_spaced_projection_angles",
    "framework_card",
    "interlaced_time_indices",
    "nyquist_demo",
    "nyquist_velocity",
    "table1_metrics_summary",
    "table_s1_ultrasparse_angles",
]
