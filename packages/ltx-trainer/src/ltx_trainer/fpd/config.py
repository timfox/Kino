"""Configuration for Fixed-Point Distillation (Wang & Tong, arXiv:2605.21484)."""

from __future__ import annotations

from dataclasses import dataclass, field

DRIFT_BANDWIDTHS: tuple[float, ...] = (0.02, 0.05, 0.2)
DINOV3_BLOCKS: tuple[int, ...] = (2, 5, 8, 11)


@dataclass
class FPDConfig:
    """Defaults from paper Sec. 3–4."""

    codebook_size: int = 1024
    embed_dim: int = 256
    sequence_length: int = 256
    mask_symbol: str = "[M]"
    # Training
    learning_rate: float = 1e-5
    batch_size_maskgit: int = 32
    batch_size_maskgen: int = 64
    lambda_gan: float = 1.0
    r_init: float = 0.95  # nearly fully masked initialization
    # Drift (Sec. 3.3)
    drift_bandwidths: tuple[float, ...] = DRIFT_BANDWIDTHS
    backbone_blocks: tuple[int, ...] = DINOV3_BLOCKS
    patch_grid: bool = True  # 4×4 spatial grid vs global pool
    # Table 1 — MaskGen-FPD GenEval overall
    geneval_overall_fpd: float = 0.45
    geneval_overall_dimo: float = 0.42
    geneval_overall_teacher_16step: float = 0.48
    # Table 2 — ImageNet FID
    fid_fpd_1step: float = 6.90
    fid_dimo_1step: float = 6.91
    fid_teacher_16step: float = 6.60
    requires_aux_score_net: bool = False
