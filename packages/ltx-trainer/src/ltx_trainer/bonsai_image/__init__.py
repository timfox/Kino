"""Bonsai Image 4B reference stub."""
from ltx_trainer.bonsai_image.config import BonsaiImageConfig
from ltx_trainer.bonsai_image.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
__all__ = ["BonsaiImageConfig", "evaluation_demo", "evaluation_smoke", "framework_card", "knowledge_card"]
