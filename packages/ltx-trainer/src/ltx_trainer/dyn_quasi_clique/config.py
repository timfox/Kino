"""Dynamic quasi-clique detection (arXiv:2606.05809)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DynQuasiCliqueConfig:
    paper_arxiv: str = "arXiv:2606.05809"
    title: str = "Detecting Large Quasi-cliques on Dynamic Networks"
    alpha_default: float = 0.8
    gamma_default: float = 0.5
    incremental_speedup_anchor: float = 207.0
    fully_dynamic_speedup_anchor: float = 21.0
