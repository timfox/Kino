"""Relightable Holoported Characters configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RelightableCharsConfig:
    name: str = "Relightable Holoported Characters"
    paper_arxiv: str = "arXiv:2512.00255"
    website: str = "https://vcai.mpi-inf.mpg.de/projects/RHC/"
    title: str = "Dynamic human relighting from sparse views (CVPR 2026 oral)"
