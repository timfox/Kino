"""SphereDiff — spherical latent MultiDiffusion (arXiv:2504.14396)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2504.14396"
PAPER_TITLE = (
    "SphereDiff: Tuning-free 360° Static and Dynamic Panorama Generation "
    "via Spherical Latent Representation"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

NUM_SPHERICAL_LATENTS = 2600
NUM_VIEW_DIRECTIONS = 89
FOV_DEGREES = 80.0
VIEW_OVERLAP = 0.6
WEIGHT_TAU = 0.5
ELEVATION_PROMPTS = (-90.0, -10.0, 0.0, 10.0, 90.0)


@dataclass
class SphereDiffConfig:
    num_latents: int = 256
    latent_dim: int = 16
    num_views: int = 8
    perspective_h: int = 32
    perspective_w: int = 32
    fov_deg: float = FOV_DEGREES
    weight_tau: float = WEIGHT_TAU
    use_dynamic_sampling: bool = True
    use_weighted_average: bool = True
    use_paper_view_schedule: bool = False
    foreground_beta_b: float = -3.0
    denoise_steps: int = 4
