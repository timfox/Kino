"""Mirage — latent spatial memory for video world models (arXiv:2606.09828)."""
from ltx_trainer.mirage.config import MirageConfig
from ltx_trainer.mirage.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
    paper_checks,
)

__all__ = [
    "MirageConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "paper_checks",
]
