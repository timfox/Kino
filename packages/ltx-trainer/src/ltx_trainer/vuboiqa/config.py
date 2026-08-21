"""VU-BOIQA — viewport-unaware blind omnidirectional IQA (arXiv:2503.06129)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2503.06129"
PAPER_DOI = "10.1145/3640344"
PAPER_TITLE = (
    "Viewport-Unaware Blind Omnidirectional Image Quality Assessment: "
    "A Flexible and Effective Paradigm"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
GITHUB_URL = "https://github.com/KangchengWu/OIQA"

# APS prior-equator (Sec. 3.2)
PE_MU_DEG = 91.3
PE_LAMBDA_DEG = 18.58
LATITUDE_DIVISION_THETA_T = 23
DEFAULT_NUM_PATCHES = 10
KAPPA_H = 0.2
KAPPA_W = 0.1
PATCH_SIZE = 224
PARAMS_M = 30.2
FLOPS_G = 40.8
NORM_IN_NORM_GAMMA = 4.0


@dataclass
class VuBoiqaConfig:
    num_patches: int = DEFAULT_NUM_PATCHES
    kappa_h: float = KAPPA_H
    kappa_w: float = KAPPA_W
    patch_size: int = PATCH_SIZE
    pe_mu_deg: float = PE_MU_DEG
    pe_lambda_deg: float = PE_LAMBDA_DEG
    theta_t: int = LATITUDE_DIVISION_THETA_T
    embed_dim: int = 64
    num_heads: int = 8
    use_pdff: bool = True
    use_hpa: bool = True
    use_pa: bool = True
    loss_gamma: float = NORM_IN_NORM_GAMMA
