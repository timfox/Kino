"""Neuro-Relational Programs (NRP) — arXiv:2606.11946."""

from ltx_trainer.nrp.config import NrpConfig, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.nrp.engine import NRP, embedded_query, execute, gated_query
from ltx_trainer.nrp.integration import framework_card, integration_bundle
from ltx_trainer.nrp.pipeline import evaluation_demo, evaluation_smoke

__all__ = [
    "NRP",
    "NrpConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "embedded_query",
    "evaluation_demo",
    "evaluation_smoke",
    "execute",
    "framework_card",
    "gated_query",
    "integration_bundle",
]
