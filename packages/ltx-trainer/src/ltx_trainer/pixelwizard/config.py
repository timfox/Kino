"""Hyperparameters for PixelWizard (Li et al., arXiv:2605.25801)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PixelWizardConfig:
    """Paper defaults (Sec. 4.1, 3.3–3.4)."""

    paper_arxiv: str = "arXiv:2605.25801"
    project_page: str = "https://wxliii.github.io/pixelwizard/"
    base_model: str = "Wan2.2-TI2V-5B"

    anchor_resolution: tuple[int, int] = (448, 256)
    resolution_2k: tuple[int, int] = (2560, 1440)
    resolution_4k: tuple[int, int] = (3840, 2144)

    anchor_tune_samples: int = 65_000
    hr_train_videos: int = 42_000
    learning_rate: float = 1e-5
    optimizer: str = "AdamW"

    num_diffusion_steps: int = 1000
    timestep_min: int = 500
    timestep_max: int = 800
    shortcut_candidates_k: int = 6
    exponential_beta: float = 0.7
    noise_span_power: float = 0.5

    hr_inference_steps: int = 4
    params_4k_b: float = 10.0
    peak_mem_4k_gb: float = 101.8
