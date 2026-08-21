"""NMM Roadmap configuration (arXiv:2605.25343)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NMMConfig:
    """Reference metadata for the NMM Roadmap survey."""

    paper_arxiv: str = "arXiv:2605.25343"
    project_page: str = "https://nmm-roadmap.github.io"
    authors_affiliation: str = "Tencent Youtu Lab et al."
