"""PrismAvatar: PMV Gaussian head avatar + subpixel lenticular display."""

from ltx_trainer.prism_avatar.benchmarks import benchmarks_bundle
from ltx_trainer.prism_avatar.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PrismAvatarConfig
from ltx_trainer.prism_avatar.mock import evaluation_smoke
from ltx_trainer.prism_avatar.paper import framework_card
from ltx_trainer.prism_avatar.pipeline import (
    run_display_smoke,
    run_marcel_metrics_smoke,
    run_training_smoke,
)

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "PrismAvatarConfig",
    "benchmarks_bundle",
    "evaluation_smoke",
    "framework_card",
    "run_display_smoke",
    "run_marcel_metrics_smoke",
    "run_training_smoke",
]
