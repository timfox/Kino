"""Dynamic consistent submodular maximization (arXiv:2606.04946)."""

from ltx_trainer.dyn_submod_cons.config import DynSubmodConsConfig
from ltx_trainer.dyn_submod_cons.mock import evaluation_smoke
from ltx_trainer.dyn_submod_cons.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.dyn_submod_cons.scheduling import (
    coverage_submodular,
    dynamic_consistent_cardinality_smoke,
    greedy_under_stream,
    random_scheduling_levels,
    toy_coverage_instance,
)

__all__ = [
    "DynSubmodConsConfig",
    "benchmarks_bundle",
    "coverage_submodular",
    "dynamic_consistent_cardinality_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "greedy_under_stream",
    "random_scheduling_levels",
    "toy_coverage_instance",
]
