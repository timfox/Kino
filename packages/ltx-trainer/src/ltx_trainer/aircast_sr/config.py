"""AirCast-SR configuration (arXiv:2605.26130)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AirCastSRConfig:
    paper_arxiv: str = "arXiv:2605.26130"
    title: str = "AirCast-SR: kilometer-scale atmospheric super-resolution via latent consistency diffusion"

    # Resolution / cadence
    input_resolution_deg: float = 0.25
    target_resolution_km: float = 1.0
    graphcast_cadence_hours: int = 6
    target_cadence_hours: int = 1
    forecast_hours: int = 67
    t_cond: int = 12  # 6-hourly GraphCast steps in conditioning window
    t_target: int = 67

    # Channels
    n_target_variables: int = 7
    n_conditioning_channels: int = 20
    denoiser_in_channels: int = 28
    denoiser_out_channels: int = 7

    # U-Net 3D (paper: 64,128,256,512 — smoke uses smaller blocks)
    block_channels: tuple[int, ...] = (64, 128, 256, 512)
    smoke_block_channels: tuple[int, ...] = (16, 32, 64)
    num_encoder_stages: int = 4
    layers_per_block: int = 2
    group_norm_groups: int = 8
    cross_attention_heads: int = 8

    # Diffusion / LCM
    training_timesteps: int = 1000
    lcm_inference_steps: int = 25
    lcm_steps_options: tuple[int, ...] = (4, 8, 25, 50)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    train_patch_size: int = 64
    infer_patch_size: int = 256
    infer_stride: int = 128

    # Training data (paper)
    train_region: str = "CONUS"
    train_year: int = 2021
    train_batch_size: int = 1

    target_variables: tuple[str, ...] = (
        "precipitation",
        "t2m",
        "q2m",
        "u10",
        "v10",
        "sp",
        "dlwrf",
    )

    case_studies: tuple[str, ...] = field(
        default_factory=lambda: (
            "2022-12-22_winter_storm_elliott",
            "2022-06-12_summer_convective",
            "2023-03-31_spring_transition",
        )
    )

    zero_shot_domains: tuple[str, ...] = ("India", "Germany")
