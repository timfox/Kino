"""TPGS — transition-plane panoramic 3DGS (arXiv:2504.09062)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2504.09062"
PAPER_TITLE = (
    "You Need a Transition Plane: Bridging Continuous Panoramic 3D "
    "Reconstruction with Perspective Gaussian Splatting"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
GITHUB_URL = "https://github.com/zhijieshen-bjtu/TPGS"

CUBE_FOV_DEG = 90.0
TRANSITION_YAW_DEG = 45.0
LAMBDA_L1 = 0.8
LAMBDA_DSSIM = 0.2
INTRA_STEPS = 10_000
INTER_STEPS = 100_000


@dataclass
class TpgsConfig:
    erp_h: int = 256
    erp_w: int = 512
    cube_face_res: int = 128
    padding_px: int = 8
    use_transition_plane: bool = True
    use_intra_inter: bool = True
    use_cube_padding: bool = True
    intra_steps: int = 100
    inter_steps: int = 50
    lambda_l1: float = LAMBDA_L1
    lambda_dssim: float = LAMBDA_DSSIM
