"""SphereDiff — tuning-free 360° panorama via spherical latents (arXiv:2504.14396)."""

from ltx_trainer.spherediff.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, SphereDiffConfig
from ltx_trainer.spherediff.mock import evaluation_smoke

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "SphereDiffConfig",
    "evaluation_smoke",
]
