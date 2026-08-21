"""uLayout — unified perspective + panoramic room layout (arXiv:2503.21562)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2503.21562"
PAPER_TITLE = (
    "uLayout: Unified Room Layout Estimation for Perspective and Panoramic Images"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
GITHUB_URL = "https://github.com/JonathanLee112/uLayout"

PANO_ERP_WIDTH = 1024
PP_ERP_WIDTH = 256  # 75% crop of informative columns (Sec. 3.3.1)
PP_PAD_WIDTH = 1024
FEATURE_HEIGHT = 1024
LAMBDA_B = 1.0
MU_DEPTH = 0.1
GAMMA_GEOM = 0.01
DELTA_PP = 1.0


@dataclass
class ULayoutConfig:
    pano_width: int = PANO_ERP_WIDTH
    pp_width: int = PP_ERP_WIDTH
    feature_h: int = 256
    pp_feature_w: int = 64
    pano_feature_w: int = 256
    use_vertical_shift: bool = True
    use_efficient_extractor: bool = True
    lambda_b: float = LAMBDA_B
    mu_depth: float = MU_DEPTH
    gamma_geom: float = GAMMA_GEOM
    delta_pp: float = DELTA_PP
    swg_repeats: int = 2
