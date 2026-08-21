"""Multi-objective submodular maximization under DP (arXiv:2606.05596)."""

from ltx_trainer.mosm_dp.algorithms import (
    dp_bicriteria_smoke,
    dp_greedy_single,
    dp_multi_greedy,
    minimax_objective,
    toy_three_objective_instance,
)
from ltx_trainer.mosm_dp.config import MosmDpConfig
from ltx_trainer.mosm_dp.mock import evaluation_smoke
from ltx_trainer.mosm_dp.pipeline import benchmarks_bundle, evaluation_demo, framework_card

__all__ = [
    "MosmDpConfig",
    "benchmarks_bundle",
    "dp_bicriteria_smoke",
    "dp_greedy_single",
    "dp_multi_greedy",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "minimax_objective",
    "toy_three_objective_instance",
]
