"""PAR — Panoramic AutoRegressive model (arXiv:2505.16862)."""

from ltx_trainer.par.config import PARConfig, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.par.mock import evaluation_smoke

__all__ = [
    "PARConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
