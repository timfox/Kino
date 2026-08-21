"""S3PO — 360° video super-resolution with WSS-L1."""

from ltx_trainer.s3po.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, S3POConfig
from ltx_trainer.s3po.mock import evaluation_smoke

__all__ = [
    "S3POConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
