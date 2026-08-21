"""One Flight Over the Gap — perspective-to-panoramic vision survey (arXiv:2509.04444)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2509.04444"
PAPER_TITLE = "One Flight Over the Gap: A Survey from Perspective to Panoramic Vision"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://insta360-research-team.github.io/Survey-of-Panorama/"

PAPERS_REVIEWED = 300
TASKS_COVERED = 20
DEFAULT_ERP_HEIGHT = 256
DEFAULT_ERP_WIDTH = 512


@dataclass
class PanoFlightConfig:
    height: int = DEFAULT_ERP_HEIGHT
    width: int = DEFAULT_ERP_WIDTH
