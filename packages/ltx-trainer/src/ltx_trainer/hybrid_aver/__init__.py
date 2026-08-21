"""Stable hybrid cross-attention AVER (arXiv:2606.03747)."""

from ltx_trainer.hybrid_aver.config import HybridAverConfig
from ltx_trainer.hybrid_aver.mock import evaluation_smoke
from ltx_trainer.hybrid_aver.pipeline import evaluation_demo, framework_card
from ltx_trainer.hybrid_aver.ltx_plan import ltx_integration_plan

__all__ = [
    "HybridAverConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
