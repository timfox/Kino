"""Configuration for SimInsert (Chen et al., arXiv:2605.23245)."""

from __future__ import annotations

from dataclasses import dataclass


BASELINES: tuple[str, ...] = (
    "pix2video",
    "fatezero",
    "consisti2v",
    "anyv2v",
)

PAPER_METRICS: tuple[str, ...] = (
    "psnr",
    "ssim",
    "lpips",
    "vfid",
    "clip_i",
    "clip_t",
)


@dataclass
class SimInsertConfig:
    """Defaults from paper Sec. III–IV."""

    num_frames: int = 49
    height: int = 480
    width: int = 854  # ~480p 16:9
    fusion_retention_p: float = 0.5  # Bernoulli(p) in Eq. 5
    use_sparse_fusion: bool = True
    use_regional_clone: bool = True
    use_latent_refresh: bool = True
    # Paper Table I (Ours row) for reference reporting
    reference_psnr: float = 36.26
    reference_ssim: float = 0.8671
    reference_lpips: float = 0.1471
    reference_vfid: float = 1062.92
    reference_clip_i: float = 0.9923
    reference_clip_t: float = 0.2825
