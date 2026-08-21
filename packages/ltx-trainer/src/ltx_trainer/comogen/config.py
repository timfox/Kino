"""Configuration for CoMoGen (Meric et al., arXiv:2605.22996)."""

from __future__ import annotations

from dataclasses import dataclass


DATASETS: tuple[str, ...] = ("clevrer", "behave")
BASELINES: tuple[str, ...] = ("goflow", "interdyn", "magicmotion")


@dataclass
class CoMoGenConfig:
    """Defaults from paper Sec. 4–5 and Appendix A."""

    num_dit_layers: int = 60
    num_motion_layers: int = 11  # top-11 layers (Sec. 4.1)
    latent_channels: int = 16
    temporal_downsample: int = 4
    spatial_downsample: int = 8
    training_resolution: tuple[int, int] = (360, 640)  # 360p
    inference_param_pct: float = 1.84
    training_param_pct: float = 3.17
    lora_rank: int = 8
    # BEHAVE reference metrics (Table 3, Ours)
    reference_ssim: float = 0.7940
    reference_psnr: float = 22.99
    reference_lpips: float = 0.0721
    reference_fvd: float = 327.63
