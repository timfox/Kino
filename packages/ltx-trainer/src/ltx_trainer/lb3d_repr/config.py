"""Learning-based 3D representations survey (Schockaert et al.; arXiv:2606.04871)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

PAPER_ARXIV = "2606.04871"
PAPER_TITLE = "Recent Advances and Trends in Learning-based 3D Representations"
PAPER_VENUE = "Computer Graphics Forum (submitted 6/2026)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_AUTHORS = (
    "Adrien Schockaert, Hamid Laga, Hazem Wannous, Vincent Magnier, "
    "Guillaume Dufaye, Jean-françois Witz"
)


class RepresentationClass(str, Enum):
    SURFACE = "surface"
    VOLUMETRIC = "volumetric"
    HYBRID = "hybrid"


class DomainNature(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class LB3DReprConfig:
    """Stub config — literature anchors only (no GPU training)."""

    seed: int = 0
    default_query_dim: int = 3


__all__ = [
    "DomainNature",
    "LB3DReprConfig",
    "PAPER_ARXIV",
    "PAPER_AUTHORS",
    "PAPER_TITLE",
    "PAPER_URL",
    "PAPER_VENUE",
    "RepresentationClass",
]
