"""CubeDiff — cubemap multi-view diffusion panoramas (ICLR 2025, arXiv:2501.17162)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2501.17162"
PAPER_TITLE = "CubeDiff: Repurposing Diffusion-Based Image Models for Panorama Generation"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PROJECT_URL = "https://cubediff.github.io/"
NUM_FACES = 6
FACE_FOV_DEG = 90.0
OVERLAP_DEG = 2.5
FACE_FOV_TRAIN_DEG = FACE_FOV_DEG + 2 * OVERLAP_DEG  # 95°
LATENT_HW = 128
TRAIN_PANORAMAS = 48000
TRAIN_ITERS = 30000


@dataclass
class CubeDiffConfig:
    num_faces: int = NUM_FACES
    face_size: int = 512
    latent_ch: int = 8
    use_synced_gn: bool = True
    use_inflated_attn: bool = True
    use_pos_encoding: bool = True
    use_overlap: bool = True
    cfg_drop: float = 0.1
