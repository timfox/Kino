"""SCOPE configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SCOPEConfig:
    name: str = "SCOPE"
    paper_arxiv: str = "arXiv:2605.23345"
    website: str = "https://z2tong.github.io/SCOPE/"
    title: str = "FPS world model with in-scope vs out-of-scope action conditioning"
