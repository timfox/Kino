"""Terastal heterogeneous multi-DNN scheduling (Wu et al., arXiv:2606.06818)."""

from ltx_trainer.terastal.config import TerastalConfig
from ltx_trainer.terastal.mock import evaluation_smoke
from ltx_trainer.terastal.paper import knowledge_bundle, paper_card
from ltx_trainer.terastal.pipeline import run_demo

__all__ = [
    "TerastalConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
