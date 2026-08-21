"""Dynamic Gaussian Processes for evolving functions (van Hulst et al., arXiv:2606.06705)."""

from ltx_trainer.dynamic_gp.config import DynamicGPConfig
from ltx_trainer.dynamic_gp.mock import evaluation_smoke
from ltx_trainer.dynamic_gp.paper import knowledge_bundle, paper_card
from ltx_trainer.dynamic_gp.pipeline import run_demo

__all__ = [
    "DynamicGPConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
