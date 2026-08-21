"""KIT IWSLT 2026 multilingual long-form speech instruction following."""

from ltx_trainer.iwslt26_if.config import IWSLT26IFConfig
from ltx_trainer.iwslt26_if.mock import evaluation_smoke
from ltx_trainer.iwslt26_if.pipeline import evaluation_demo, framework_card
from ltx_trainer.iwslt26_if.ltx_plan import ltx_integration_plan

__all__ = [
    "IWSLT26IFConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
