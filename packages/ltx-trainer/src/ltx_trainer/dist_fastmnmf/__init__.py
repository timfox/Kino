"""Distributed FastMNMF — block-diagonal SCM BSS for distributed arrays (arXiv:2605.19388)."""

from ltx_trainer.dist_fastmnmf.block_scm import block_diag_scm, per_subarray_joint_diagonalizable, subarray_mic_index
from ltx_trainer.dist_fastmnmf.complexity import asymptotic_speedup_equal_subarrays, complexity_table_rows, relative_runtime_vs_all
from ltx_trainer.dist_fastmnmf.config import DistFastmnmfConfig, SdrResults, TimingResults
from ltx_trainer.dist_fastmnmf.layout import LIMITATIONS
from ltx_trainer.dist_fastmnmf.mock import evaluation_smoke
from ltx_trainer.dist_fastmnmf.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    experiment_setup,
    framework_card,
    headline_results,
    sdr_results,
    table_ii_computation_time,
)

__all__ = [
    "LIMITATIONS",
    "DistFastmnmfConfig",
    "SdrResults",
    "TimingResults",
    "asymptotic_speedup_equal_subarrays",
    "benchmarks_bundle",
    "block_diag_scm",
    "complexity_table_rows",
    "evaluation_demo",
    "evaluation_smoke",
    "experiment_setup",
    "framework_card",
    "headline_results",
    "per_subarray_joint_diagonalizable",
    "relative_runtime_vs_all",
    "sdr_results",
    "subarray_mic_index",
    "table_ii_computation_time",
]
