"""Dense360 — omnidirectional dense VLM understanding."""

from ltx_trainer.dense360.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, Dense360Config
from ltx_trainer.dense360.mock import evaluation_smoke

__all__ = [
    "Dense360Config",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
