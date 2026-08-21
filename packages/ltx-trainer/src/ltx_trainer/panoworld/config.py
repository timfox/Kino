"""PanoWorld configuration (Wang et al. arXiv:2605.13169)."""

from __future__ import annotations

from dataclasses import dataclass, field

PAPER_ARXIV = "2605.13169"
PAPER_TITLE = "PanoWorld: Towards Spatial Supersensing in 360° Panorama World"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://wcpcp.github.io/PanoWorld"

# Four capability families (Sec. 3.2)
CAPABILITY_FAMILIES = (
    "semantic_anchoring",
    "spherical_grounding",
    "reference_frame_transformation",
    "depth_aware_3d",
)

# PanoSpace-Bench task categories (Table 12)
PANOSPACE_CATEGORIES = (
    "absolute_direction",
    "bfov_localization",
    "relative_direction",
    "camera_rotation",
    "object_reorientation",
    "observer_distance",
    "relative_3d",
    "seam_continuity",
)

CORPUS_SIZE = 570_321
INSTRUCTION_CANONICAL = 2_997_516


@dataclass
class PanoWorldConfig:
    height: int = 160
    width: int = 320
    patch_size: int = 16
    hidden_dim: int = 64
    num_heads: int = 4
    spherical_freq_bands: int = 8
    gate_init: float = 0.01
    num_choices: int = 4
    max_entities: int = 32
    train_lr: float = 1e-6
    train_weight_decay: float = 0.01
    grad_clip: float = 1.0
    capability_weights: dict[str, float] = field(
        default_factory=lambda: {
            "semantic_anchoring": 0.25,
            "spherical_grounding": 0.30,
            "reference_frame_transformation": 0.25,
            "depth_aware_3d": 0.20,
        }
    )
