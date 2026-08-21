"""Bonsai Image 4B configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class BonsaiImageConfig:
    name: str = "Bonsai Image 4B"
    paper_arxiv: str = ""
    website: str = "https://prismml.com/news/bonsai-image-4b"
    title: str = "1-bit/ternary FLUX.2 Klein 4B for on-device image generation"
