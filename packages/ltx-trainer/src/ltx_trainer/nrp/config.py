"""Neuro-Relational Programs configuration (Soeteman et al., arXiv:2606.11946)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.11946"
PAPER_TITLE = "Neuro-Relational Programs: Unifying Queries and Neural Computation over Structured Data"
PAPER_AUTHORS = (
    "Arie Soeteman, Balder ten Cate, Maurice Funk, Benny Kimelfeld, "
    "Carsten Lutz, Moritz Schönherr"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "Preprint Jun 2026"
PAPER_IMPL_REF = "https://arxiv.org/abs/2605.24207"


@dataclass
class NrpConfig:
    default_embedding_dim: int = 4
    relu_hidden: int = 8
    acceptance_positive_min: bool = True
