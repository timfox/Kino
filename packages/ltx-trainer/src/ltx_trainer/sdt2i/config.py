"""SDT2I — spherical dense text-to-image (arXiv:2502.12691)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2502.12691"
PAPER_TITLE = "Spherical Dense Text-to-Image Synthesis"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PANORAMA_TRIGGER = "360-degree panoramic image"
ERP_W = 1024
ERP_H = 512
DEFAULT_BOOTSTRAP = 20
DEFAULT_STRIDE = 8


@dataclass
class Sdt2iConfig:
    erp_w: int = ERP_W
    erp_h: int = ERP_H
    latent_ch: int = 4
    bootstrap_steps: int = DEFAULT_BOOTSTRAP
    stride: int = DEFAULT_STRIDE
    use_stitch: bool = True
    lora_foreground: bool = False
    lora_background: bool = True
    md_branch: str = "pano"  # pano | pers | both
    bootstrap_coupling_branches: bool = False
    bootstrap_coupling_objects: bool = False
    fg_eppa: bool = True
    global_prompt: bool = False
    erp_mask_reproj: bool = True
