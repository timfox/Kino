"""GPU midsize multi-precision integer division (Marchioro et al., arXiv:2606.06386)."""

from ltx_trainer.midint_division.config import MidintDivisionConfig
from ltx_trainer.midint_division.division import divide
from ltx_trainer.midint_division.mock import evaluation_smoke
from ltx_trainer.midint_division.paper import knowledge_bundle, paper_card
from ltx_trainer.midint_division.pipeline import run_demo

__all__ = [
    "MidintDivisionConfig",
    "divide",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
