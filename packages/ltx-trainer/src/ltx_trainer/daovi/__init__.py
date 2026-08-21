"""DAOVI — distortion-aware omnidirectional video inpainting."""

from ltx_trainer.daovi.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, DaoviConfig
from ltx_trainer.daovi.mock import evaluation_smoke

__all__ = [
    "DaoviConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
