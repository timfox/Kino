"""MOSM under differential privacy (arXiv:2606.05596)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MosmDpConfig:
    paper_arxiv: str = "arXiv:2606.05596"
    title: str = "Multi-Objective Submodular Maximization with Differential Privacy"
    epsilon_default: float = 1.0
    delta_default: float = 1e-5
    d_max_objectives: int = 8
    requires_d_le_k: bool = True
