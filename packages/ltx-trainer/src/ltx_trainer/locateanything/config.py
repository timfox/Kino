"""LocateAnything configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class LocateAnythingConfig:
    name: str = "LocateAnything"
    paper_arxiv: str = "arXiv:2605.27365"
    website: str = "https://research.nvidia.com/labs/lpr/locate-anything/"
    title: str = "Fast VLM grounding via Parallel Box Decoding (PBD)"
