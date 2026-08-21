"""Entity-Aware CoT for speech LLM entity binding (arXiv:2606.04474)."""

from ltx_trainer.ea_cot.config import EACoTConfig
from ltx_trainer.ea_cot.mock import evaluation_smoke
from ltx_trainer.ea_cot.pipeline import evaluation_demo, framework_card
from ltx_trainer.ea_cot.ltx_plan import ltx_integration_plan

__all__ = [
    "EACoTConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
