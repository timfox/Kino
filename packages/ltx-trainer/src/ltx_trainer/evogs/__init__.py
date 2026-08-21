"""EvoGS — continuous-layered Gaussian splatting with evolution tree (arXiv:2606.07179)."""

from ltx_trainer.evogs.benchmarks import benchmarks_bundle, ours_beats_baselines, table3_ghost_splat_ratios
from ltx_trainer.evogs.config import EvoGSConfig, EvoGSParams, RefinementMode
from ltx_trainer.evogs.layout import LIMITATIONS
from ltx_trainer.evogs.ltx_plan import gopex_env_snippet, ltx_integration_plan
from ltx_trainer.evogs.pipeline import evaluation_demo, evaluation_smoke, framework_card, paper_limitations
from ltx_trainer.evogs.refinement import reconstruct_leaf, split_children
from ltx_trainer.evogs.simulation import full_pipeline_demo, train_step_torch
from ltx_trainer.evogs.tree import EvolutionTree, synthetic_tree

__all__ = [
    "EvolutionTree",
    "EvoGSConfig",
    "EvoGSParams",
    "LIMITATIONS",
    "RefinementMode",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "full_pipeline_demo",
    "gopex_env_snippet",
    "ltx_integration_plan",
    "ours_beats_baselines",
    "paper_limitations",
    "reconstruct_leaf",
    "split_children",
    "synthetic_tree",
    "table3_ghost_splat_ratios",
    "train_step_torch",
]
