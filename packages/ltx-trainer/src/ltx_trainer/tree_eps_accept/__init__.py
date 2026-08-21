"""Tree automata ε-acceptance stub (arXiv:2605.27192)."""

from ltx_trainer.tree_eps_accept.config import TreeEpsAcceptConfig
from ltx_trainer.tree_eps_accept.distance import bd_at_roots, bisimulation_distance
from ltx_trainer.tree_eps_accept.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.tree_eps_accept.measure import defect_ok_mass, leaf_mass_from, relative_measure
from ltx_trainer.tree_eps_accept.mock import evaluation_smoke
from ltx_trainer.tree_eps_accept.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.tree_eps_accept.tables import headline_results
from ltx_trainer.tree_eps_accept.trees import LabelledTree, full_binary_tree

__all__ = [
    "LIMITATIONS",
    "PIPELINE_STAGES",
    "LabelledTree",
    "TreeEpsAcceptConfig",
    "benchmarks_bundle",
    "bd_at_roots",
    "bisimulation_distance",
    "defect_ok_mass",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "full_binary_tree",
    "headline_results",
    "leaf_mass_from",
    "relative_measure",
]
