"""LALE remote sensing segmentation (Çağlar & Temizel, arXiv:2606.02092)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LaleConfig:
    paper_arxiv: str = "arXiv:2606.02092"
    title: str = (
        "LALE: Lightweight-Transformer Architecture for Land-Cover Estimation"
    )
    dataset: str = "ARAS400k"
    dataset_ref: str = "arXiv:2603.09625"  # ARAS400k grounding paper

    # Input / stem (§3.1)
    image_size: int = 256
    stem_channels: int = 32
    stem_kernel: int = 3
    stem_stride: int = 2

    # Four-stage encoder channel dims
    stage_channels: tuple[int, ...] = (32, 64, 128, 256)
    conv_mixer_stages: tuple[int, ...] = (1, 2)
    transformer_stages: tuple[int, ...] = (3, 4)

    # Decoder (§3.4)
    decoder_channels: int = 128
    decoder_dropout: float = 0.1
    num_classes: int = 8  # ARAS400k land-cover classes (benchmark default)

    # Model scales S1–S4 (Table 2)
    scales: tuple[str, ...] = ("S1", "S2", "S3", "S4")

    # Headline vs UPerNet (abstract / Table 1)
    upernet_f1: float = 77.31
    lale_s1_f1: float = 74.69
    lale_s1_params_m: float = 1.6
    lale_s1_gmacs: float = 0.59
    upernet_params_m: float = 11.6
    upernet_gmacs: float = 13.62
    f1_gap_to_upernet: float = 2.6  # within 2.6 F1 points

    # Training (§4)
    loss: str = "Dice"
    train_images: int = 80192
    train_hours_per_model_h100: float = 2.2
