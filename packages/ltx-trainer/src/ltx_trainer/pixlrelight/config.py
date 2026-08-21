"""PIXLRelight configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PIXLRelightConfig:
    name: str = "PIXLRelight"
    paper_arxiv: str = "arXiv:2605.18735"
    website: str = "https://mlfarinha.github.io/pixl-relight/"
    title: str = "PBR-controllable single-image relighting via intrinsic conditioning"
