"""LLM C++→CUDA Deopt-Reopt workflow (Mukunoki et al., arXiv:2606.06063)."""

from ltx_trainer.deopt_reopt.config import DeoptReoptConfig
from ltx_trainer.deopt_reopt.mock import evaluation_smoke
from ltx_trainer.deopt_reopt.paper import knowledge_bundle, paper_card
from ltx_trainer.deopt_reopt.pipeline import run_demo

__all__ = [
    "DeoptReoptConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
