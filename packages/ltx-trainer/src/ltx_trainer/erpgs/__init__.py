"""ErpGS — equirectangular 3D Gaussian splatting (arXiv:2505.19883)."""

from ltx_trainer.erpgs.config import ErpGSConfig, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.erpgs.mock import evaluation_smoke

__all__ = [
    "ErpGSConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
