"""SPIM Equilibrium Propagation — hybrid optical EP (arXiv:2606.13454)."""

from ltx_trainer.spim_ep.config import (
    FINITE_DIFF_DELTA,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PatternMode,
    SPIMEPConfig,
)
from ltx_trainer.spim_ep.paper import evaluation_demo, framework_card

__all__ = [
    "FINITE_DIFF_DELTA",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "PatternMode",
    "SPIMEPConfig",
    "evaluation_demo",
    "framework_card",
]
