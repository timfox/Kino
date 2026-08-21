"""Human-view video understanding survey configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.humanview_vu.constants import (
    AWESOME_URL,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    TABLE1_AXES,
)


@dataclass
class HumanViewVuConfig:
    paper_arxiv: str = PAPER_ARXIV
    paper_title: str = PAPER_TITLE
    paper_url: str = PAPER_URL
    awesome_url: str = AWESOME_URL
    default_frames: int = 32
    table1_axes: tuple[str, ...] = TABLE1_AXES


__all__ = [
    "AWESOME_URL",
    "HumanViewVuConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "TABLE1_AXES",
]
