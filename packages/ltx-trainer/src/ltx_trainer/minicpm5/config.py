"""MiniCPM5-1B configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class MiniCPM5Config:
    name: str = "MiniCPM5-1B"
    paper_arxiv: str = ""
    website: str = "https://huggingface.co/openbmb/MiniCPM5-1B"
    title: str = "OpenBMB MiniCPM5 1B edge multimodal LM"
