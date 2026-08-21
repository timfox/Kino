"""(k, δ)-truss configuration (Hu et al., arXiv:2606.11582)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.11582"
PAPER_TITLE = (
    "Querying Cohesive Subgraph regarding Span-Constrained Triangles "
    "on Temporal Graphs with Dynamic Index Maintenance"
)
PAPER_AUTHORS = "Chuhan Hu, Ming Zhong, Lei Li"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "ICDE 2024 / arXiv Jun 2026"
PAPER_REPO = PAPER_URL

# Default query parameters used in experiments (Sec. VII-B)
DEFAULT_K_FRAC = 0.30
DEFAULT_DELTA_FRAC = 0.60


@dataclass
class KdTrussConfig:
    k: int = 4
    delta: int = 1
    use_tc_index: bool = True
    use_dc_index: bool = False
    maintenance_local_search: bool = True
