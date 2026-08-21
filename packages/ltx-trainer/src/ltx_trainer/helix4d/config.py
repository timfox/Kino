"""Helix4D video-to-4D mesh generation (Yenphraphai et al., arXiv:2605.26109)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Helix4DConfig:
    """Defaults from paper Sec. 4–5."""

    num_frames: int = 16
    voxel_grid: int = 64
    rope_theta: float = 10_000.0
    rope_alpha: float = 0.75
    attention_window: int = 5
    head_dim: int = 64
    train_iters: int = 20_000
    batch_size: int = 32
    learning_rate: float = 2e-5

    stages: tuple[str, ...] = field(
        default_factory=lambda: ("sparse_structure", "geometry", "material")
    )
    baselines: tuple[str, ...] = field(
        default_factory=lambda: (
            "ss4d",
            "shapegen4d",
            "mesh4d",
            "motion_3_to_4",
            "actionmesh",
        )
    )
