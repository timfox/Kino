"""CubePart configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CubePartConfig:
    name: str = "CubePart"
    paper_arxiv: str = "arXiv:2605.28763"
    website: str = "https://cubepart.github.io/"
    title: str = "Open-vocabulary part-controllable 3D mesh (SIGGRAPH 2026)"
