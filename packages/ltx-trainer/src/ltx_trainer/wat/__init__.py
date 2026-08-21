"""Wavelet as Tokenizer (WAT) — arXiv:2606.02631."""

from ltx_trainer.wat.config import WATConfig
from ltx_trainer.wat.mock import evaluation_smoke
from ltx_trainer.wat.pipeline import evaluation_demo, framework_card
from ltx_trainer.wat.ltx_plan import ltx_integration_plan

__all__ = [
    "WATConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
