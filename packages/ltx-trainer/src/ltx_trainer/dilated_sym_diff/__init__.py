"""Dilated symmetric difference for binary image comparison (Urieli, arXiv:2606.06512)."""

from ltx_trainer.dilated_sym_diff.config import DilatedSymDiffConfig
from ltx_trainer.dilated_sym_diff.mock import evaluation_smoke
from ltx_trainer.dilated_sym_diff.paper import knowledge_bundle, paper_card
from ltx_trainer.dilated_sym_diff.pipeline import run_demo

__all__ = [
    "DilatedSymDiffConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
