"""AnthropoCam Anthropocene mobile NST (arXiv:2601.21141)."""

from ltx_trainer.anthropocam.config import AnthropoCamConfig
from ltx_trainer.anthropocam.mock import evaluation_smoke
from ltx_trainer.anthropocam.pipeline import evaluation_demo, framework_card, knowledge_blob
from ltx_trainer.anthropocam.ltx_plan import ltx_integration_plan

__all__ = [
    "AnthropoCamConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_blob",
    "ltx_integration_plan",
]
