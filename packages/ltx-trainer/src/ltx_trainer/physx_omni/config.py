"""PhysX-Omni configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PhysXOmniConfig:
    name: str = "PhysX-Omni"
    paper_arxiv: str = "arXiv:2605.21572"
    website: str = "https://physx-omni.github.io/"
    title: str = "Simulation-ready rigid/deformable/articulated 3D generation"
