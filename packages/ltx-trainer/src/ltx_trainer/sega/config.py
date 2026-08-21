"""SEGA configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SEGAConfig:
    name: str = "SEGA"
    paper_arxiv: str = "arXiv:2605.22668"
    website: str = "https://rajabi2001.github.io/sega/"
    title: str = "Spectral-energy guided RoPE scaling for DiT resolution extrapolation"
