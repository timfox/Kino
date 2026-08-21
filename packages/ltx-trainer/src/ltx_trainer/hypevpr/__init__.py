"""HypeVPR — hyperbolic P2E visual place recognition."""

from ltx_trainer.hypevpr.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, HypeVPRConfig
from ltx_trainer.hypevpr.mock import evaluation_smoke

__all__ = [
    "HypeVPRConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
