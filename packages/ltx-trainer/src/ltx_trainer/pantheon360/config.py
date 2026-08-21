"""Pantheon360 configuration (arXiv:2605.25449)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pantheon360Config:
    """Reference hyperparameters from Pantheon360."""

    paper_arxiv: str = "arXiv:2605.25449"
    project_page: str = "https://koi953215.github.io/pantheon360_page/"
    erp_height: int = 512
    erp_width: int = 1024
    num_frames: int = 25
    clip_yaw_crops: int = 8
    yaw_step_deg: float = 45.0
    pi3_confidence_threshold: float = 0.25
    guidance_scale: float = 5.0
    denoise_steps: int = 25
    recon_backend: str = "PI3"
