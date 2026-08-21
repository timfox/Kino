"""Branch-level energy localization (Montoya et al., arXiv:2606.07076)."""

from ltx_trainer.branch_energy.classical import classical_card, classical_comparisons
from ltx_trainer.branch_energy.config import BranchEnergyConfig
from ltx_trainer.branch_energy.mock import evaluation_smoke
from ltx_trainer.branch_energy.paper import knowledge_bundle, paper_card
from ltx_trainer.branch_energy.pipeline import run_demo

__all__ = [
    "BranchEnergyConfig",
    "classical_card",
    "classical_comparisons",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
