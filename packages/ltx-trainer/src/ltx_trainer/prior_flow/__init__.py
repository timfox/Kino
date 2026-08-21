"""PriOr-Flow — panoramic optical flow with orthogonal-view prior."""

from ltx_trainer.prior_flow.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PriorFlowConfig
from ltx_trainer.prior_flow.mock import evaluation_smoke

__all__ = [
    "PriorFlowConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
