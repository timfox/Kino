"""ChWDTA learned image compression (arXiv:2606.00111)."""

from ltx_trainer.chwdta.config import ChwdtaConfig
from ltx_trainer.chwdta.mock import evaluation_smoke
from ltx_trainer.chwdta.pipeline import evaluation_demo, framework_card
from ltx_trainer.chwdta.ltx_plan import ltx_integration_plan

__all__ = [
    "ChwdtaConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
