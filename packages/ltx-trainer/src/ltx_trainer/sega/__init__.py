"""SEGA reference stub."""
from ltx_trainer.sega.config import SEGAConfig
from ltx_trainer.sega.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
__all__ = ["SEGAConfig", "evaluation_demo", "evaluation_smoke", "framework_card", "knowledge_card"]
