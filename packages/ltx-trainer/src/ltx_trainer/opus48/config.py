"""Claude Opus 4.8 configuration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Opus48Config:
    name: str = "Claude Opus 4.8"
    paper_arxiv: str = ""
    website: str = "https://www.anthropic.com/news/claude-opus-4-8"
    title: str = "Anthropic frontier model release (agent coding tier)"
