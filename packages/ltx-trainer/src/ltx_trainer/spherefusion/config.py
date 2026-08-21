"""SphereFusion — panorama depth via gated ERP+sphere fusion (arXiv:2502.05859)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2502.05859"
PAPER_TITLE = "SphereFusion: Efficient Panorama Depth Estimation via Gated Fusion"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PANORAMA_H = 512
PANORAMA_W = 1024
INFERENCE_SEC_512 = 0.01741
FPS_512 = 60.0
BERHU_T = 0.2


@dataclass
class SphereFusionConfig:
    in_ch: int = 3
    erp_ch: tuple[int, ...] = (64, 256, 512, 1024, 2048)
    sph_ch: tuple[int, ...] = (64, 64, 128, 256, 512)
    mesh_mr: int = 5
    use_gatefuse: bool = True
    faf_cache: bool = True
