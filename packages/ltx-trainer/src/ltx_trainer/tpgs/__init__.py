"""TPGS — transition-plane panoramic 3D Gaussian splatting (arXiv:2504.09062)."""

from ltx_trainer.tpgs.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, TpgsConfig
from ltx_trainer.tpgs.mock import evaluation_smoke

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "TpgsConfig",
    "evaluation_smoke",
]
