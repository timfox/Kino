"""LightHarmony3D configuration (Huang et al. arXiv:2603.29209)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://arxiv.org/abs/2603.29209"
PAPER_TITLE = "LightHarmony3D: Harmonizing Illumination and Shadows for Object Insertion in 3D Gaussian Splatting"

# HDR fusion (Sec. 3.4)
FUSION_SAT_THRESHOLD = 0.9
FUSION_GAMMA = 2.4
RGB_LUMA_WEIGHTS = (0.21267, 0.71516, 0.07217)
DEFAULT_EV_SEQUENCE = (-6, -3, 0)

# GenEnvLighting (Sec. 3.3)
GENENV_TARGET_EV = -3
GENENV_LORA_RANK = 16
GENENV_TRAINING_HDRIS = 800

# Shadow shaping (Sec. 3.6)
SHADOW_GAMMA = 0.65
SHADOW_SMIN = 0.05
SHADOW_LAMBDA = 0.85
SHADOW_EPS = 1e-3
SHADOW_MIN_R0 = 0.02

# Ray-decoupled (Sec. 3.5)
RAY_TYPES_SHADOW = frozenset({"shadow", "diffuse"})
RAY_TYPES_TRANSPARENT = frozenset({"camera", "transmission", "glossy"})


@dataclass(frozen=True)
class LightHarmony3DConfig:
    image_size: int = 64
    panorama_h: int = 32
    panorama_w: int = 64
    use_gen_env: bool = True
    use_hdr_fusion: bool = True
    use_ray_decoupled: bool = True
    use_shadow_ratio: bool = True
    shadow_gamma: float = SHADOW_GAMMA
    shadow_smin: float = SHADOW_SMIN
    shadow_lambda: float = SHADOW_LAMBDA
    ev_sequence: tuple[int, ...] = DEFAULT_EV_SEQUENCE
