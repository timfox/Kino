"""Fast multirate encoding for 360° OMAF streaming (arXiv:2601.17568)."""

from ltx_trainer.oma_fme.config import OmaFmeConfig, PAPER_ARXIV, PAPER_TITLE
from ltx_trainer.oma_fme.mock import evaluation_smoke
from ltx_trainer.oma_fme.paper import paper_knowledge
from ltx_trainer.oma_fme.pipeline import run_pipeline

__all__ = [
    "OmaFmeConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "evaluation_smoke",
    "paper_knowledge",
    "run_pipeline",
]
