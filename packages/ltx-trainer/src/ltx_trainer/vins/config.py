"""Configuration for VINS-120K / UHR post-adaptation (arXiv:2605.23518)."""

from __future__ import annotations

from dataclasses import dataclass


EDIT_CATEGORIES: tuple[str, ...] = (
    "local_editing",
    "global_editing",
    "camera_movement",
    "personalized_generation",
)

EDIT_TYPES: tuple[str, ...] = (
    "addition",
    "deletion",
    "replacement",
    "color",
    "material",
    "action",
    "text",
    "tone",
    "background",
    "style",
    "translation",
    "zooming",
    "subject_driven",
)


@dataclass
class VINSConfig:
    """Defaults from paper Sec. 4–5 and Supplementary B."""

    uhr_size: int = 4096
    nhr_size: int = 1024
    lora_rank: int = 32
    learning_rate: float = 5e-6
    ffs_lambda: float = 1.0
    ffs_gamma: float = 2.0
    ffs_alpha_min: float = 0.2
    ffs_alpha_max: float = 1.2
    ffs_epsilon: float = 1e-8
    filter_retain_fraction: float = 0.2
    dataset_size: int = 120_000
    eval_size: int = 509
    avg_width: int = 4656
    avg_height: int = 4138
