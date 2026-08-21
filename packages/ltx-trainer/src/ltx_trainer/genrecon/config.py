"""GenRecon configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GenReconConfig:
    name: str = "GenRecon"
    paper_arxiv: str = "arXiv:2605.23888"
    website: str = "https://kasothaphie.github.io/GenRecon/"
    title: str = "Multi-view scene PBR mesh via Trellis.2 prior + projection conditioning"
