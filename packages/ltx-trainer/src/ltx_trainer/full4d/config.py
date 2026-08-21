"""Full-4D configuration (arXiv:2605.25500)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Full4DConfig:
    """Reference hyperparameters from the Full-4D paper."""

    paper_arxiv: str = "arXiv:2605.25500"
    project_page: str = "https://ccxi1008.github.io/Full-4D/"
    num_views: int = 6
    num_scenes: int = 2000
    num_subjects: int = 6000
    train_clips: int = 60000
    test_scenes: int = 500
    clip_frames: int = 81
    lambda_fmd: float = 1.0
    max_gaussians: int = 120000
    token_dim: int = 64
