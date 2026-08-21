"""MiniCPM5-1B reference stub."""
from ltx_trainer.minicpm5.config import MiniCPM5Config
from ltx_trainer.minicpm5.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
__all__ = ["MiniCPM5Config", "evaluation_demo", "evaluation_smoke", "framework_card", "knowledge_card"]
