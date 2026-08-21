"""NAS-VAR MRI reconstruction constants (arXiv:2605.19354)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NasVarConfig:
    paper_arxiv: str = "2605.19354"
    paper_title: str = (
        "Next-Acceleration-Scale Prediction for Autoregressive MRI Reconstruction"
    )
    institution: str = "Johns Hopkins University"

    acceleration_scales: tuple[int, ...] = (32, 16, 8, 4, 2)
    inference_scale: int = 32

    codebook_size: int = 4096
    latent_dim: int = 32
    channel_width: int = 160

    token_grids_per_scale: tuple[int, ...] = (11, 12, 13, 14, 15, 16)

    transformer_depth: int = 16
    transformer_embed_dim: int = 1024
    transformer_heads: int = 16
    transformer_mlp_ratio: float = 4.0

    eval_acceleration: int = 32
    benchmark: str = "fastMRI"
    contrasts: tuple[str, ...] = ("T1", "T2", "FLAIR")
    sampling_masks: tuple[str, ...] = (
        "ES_Cartesian_X",
        "ES_Cartesian_Y",
        "Radial",
        "Gaussian_VD",
    )

    aq_vae_loss_weights: dict[str, float] | None = None

    training_base_epochs: int = 250
    training_distill_epochs: int = 100
    inference_seconds_per_image_a5000: float = 0.2441

    def __post_init__(self) -> None:
        if self.aq_vae_loss_weights is None:
            self.aq_vae_loss_weights = {
                "LSSIM": 1.0,
                "Ladv": 0.1,
                "Lperc": 0.1,
                "Lcom": 0.25,
            }
