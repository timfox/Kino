"""URNG / UG configuration (Liang et al., arXiv:2606.11789)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.11789"
PAPER_TITLE = "Efficient Graph Indexing for Interval-Aware Vector Search"
PAPER_AUTHORS = (
    "Siyuan Liang, Ziqi Yin, Qi Zhang, Ronghua Li, Guoren Wang, "
    "Kaiwen Xue, Daiyin Wang, Xubin Li"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "Preprint Jun 2026"
PAPER_REPO = "https://github.com/bbtsrr/UG-submit"

# Default UG construction (Sec. 5.1)
DEFAULT_EF_SPATIAL = 128
DEFAULT_EF_ATTRIBUTE = 300
DEFAULT_MAX_EDGES_IF = 256
DEFAULT_MAX_EDGES_IS = 256
DEFAULT_ITERATIONS = 5


@dataclass
class UrngConfig:
    ef_spatial: int = DEFAULT_EF_SPATIAL
    ef_attribute: int = DEFAULT_EF_ATTRIBUTE
    max_edges_if: int = DEFAULT_MAX_EDGES_IF
    max_edges_is: int = DEFAULT_MAX_EDGES_IS
    iterations: int = DEFAULT_ITERATIONS
    beam_size: int = 64
    query_k: int = 10
