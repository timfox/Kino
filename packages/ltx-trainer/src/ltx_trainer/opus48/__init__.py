"""Claude Opus 4.8 reference stub."""
from ltx_trainer.opus48.config import Opus48Config
from ltx_trainer.opus48.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
__all__ = ["Opus48Config", "evaluation_demo", "evaluation_smoke", "framework_card", "knowledge_card"]
