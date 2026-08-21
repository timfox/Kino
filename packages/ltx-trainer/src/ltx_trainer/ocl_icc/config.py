"""ICC video OCL configuration (arXiv:2605.30211)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCLICCConfig:
    paper_arxiv: str = "arXiv:2605.30211"
    num_slots: int = 7
    slot_dim: int = 64
    feature_dim: int = 64
    spatial_h: int = 8
    spatial_w: int = 8
    slot_attn_iters: int = 3
    transition_noise_std: float = 0.1
    encoder_backbone: str = "DINOv2-ViT-S/14"
    input_resolution: int = 224
    basis_methods: tuple[str, ...] = ("RandSF.Q", "SmoothSA", "SlotContrast", "VideoSAUR")
    datasets: tuple[str, ...] = ("MOVi-C", "MOVi-E", "YTVIS-HQ")
