"""LALE land-cover segmentation (arXiv:2606.02092)."""

from ltx_trainer.lale.config import LaleConfig
from ltx_trainer.lale.mock import evaluation_smoke
from ltx_trainer.lale.model import LALE
from ltx_trainer.lale.pipeline import evaluation_demo, framework_card
from ltx_trainer.lale.ltx_plan import ltx_integration_plan

__all__ = [
    "LALE",
    "LaleConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
