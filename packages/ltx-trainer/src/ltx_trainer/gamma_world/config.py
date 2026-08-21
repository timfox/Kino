"""Gamma-World configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GammaWorldConfig:
    name: str = "Gamma-World"
    paper_arxiv: str = "arXiv:2605.28816"
    website: str = "https://research.nvidia.com/labs/sil/projects/gamma-world/"
    title: str = "Multi-agent generative world model (simplex RoPE + sparse hub attention)"
