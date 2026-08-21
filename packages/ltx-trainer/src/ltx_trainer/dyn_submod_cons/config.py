"""Dynamic consistent submodular maximization (arXiv:2606.04946)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DynSubmodConsConfig:
    paper_arxiv: str = "arXiv:2606.04946"
    title: str = "A General Framework for Dynamic Consistent Submodular Maximization"
    cardinality_approx: str = "1/2 - O(ε)"
    cardinality_consistency: str = "O(1/ε²)"
    matroid_approx: str = "1/4 - O(ε)"
    matroid_consistency: str = "O(log k / ε²)"
