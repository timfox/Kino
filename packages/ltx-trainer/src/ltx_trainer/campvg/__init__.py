"""CamPVG — camera-controlled panoramic video (arXiv:2509.19979)."""

from ltx_trainer.campvg.config import CamPVGConfig, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.campvg.mock import evaluation_smoke

__all__ = [
    "CamPVGConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
