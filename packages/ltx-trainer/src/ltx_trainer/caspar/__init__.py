"""Caspar — CUDA symbolic programming + GPU nonlinear solver (arXiv:2605.30583)."""

from ltx_trainer.caspar.config import CasparConfig
from ltx_trainer.caspar.dabseg import DabsegGraph, build_sin_cos_sum_graph
from ltx_trainer.caspar.factors import (
    Camera,
    Observation,
    snavely_reprojection_residual,
    stack_bal_residuals,
)
from ltx_trainer.caspar.memory import AccessPattern, SharedIndex, blocked_struct_of_arrays, build_shared_index
from ltx_trainer.caspar.metrics import (
    bal_dataset_catalog,
    benchmarks_bundle,
    table_bal_final_mse_px,
    table_bal_memory_gb,
    table_bal_speedup_vs_ceres,
    table_kernel_fusion,
    table_symbolic_optimizations,
)
from ltx_trainer.caspar.optimize import common_partial_cse, context_aware_reciprocal_mul, hardware_map_norm
from ltx_trainer.caspar.pipeline import evaluation_demo, evaluation_smoke, framework_card, paper_limitations, symbolic_demo
from ltx_trainer.caspar.reorder import estimate_register_pressure, reorder_calls
from ltx_trainer.caspar.solver import SolverStats, lm_solve, pcgnr

__all__ = [
    "AccessPattern",
    "Camera",
    "CasparConfig",
    "DabsegGraph",
    "Observation",
    "SharedIndex",
    "SolverStats",
    "bal_dataset_catalog",
    "benchmarks_bundle",
    "blocked_struct_of_arrays",
    "build_shared_index",
    "build_sin_cos_sum_graph",
    "common_partial_cse",
    "context_aware_reciprocal_mul",
    "estimate_register_pressure",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "hardware_map_norm",
    "lm_solve",
    "paper_limitations",
    "pcgnr",
    "reorder_calls",
    "snavely_reprojection_residual",
    "stack_bal_residuals",
    "symbolic_demo",
    "table_bal_final_mse_px",
    "table_bal_memory_gb",
    "table_bal_speedup_vs_ceres",
    "table_kernel_fusion",
    "table_symbolic_optimizations",
]
