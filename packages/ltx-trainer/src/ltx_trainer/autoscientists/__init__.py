"""AUTOSCIENTISTS: self-organizing agent teams for long-running science (arXiv:2605.28655)."""

from ltx_trainer.autoscientists.config import AutoScientistsConfig
from ltx_trainer.autoscientists.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
from ltx_trainer.autoscientists.workflow import run_execution_cycle

__all__ = [
    "AutoScientistsConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "run_execution_cycle",
]
