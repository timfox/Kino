"""Hyperparameters for MSFET-E2V (Maqsood et al., arXiv:2605.25804)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MSFETE2VConfig:
    """Paper defaults (Sec. III–IV)."""

    paper_arxiv: str = "arXiv:2605.25804"
    title: str = "MSFET-E2V: multiscale frequency-enhanced transformer for E2V"

    sensor_resolution: tuple[int, int] = (240, 180)
    """DAVIS240C training / ECD resolution (H, W)."""

    voxel_bins: int = 5
    base_channels: int = 32
    embed_dim: int = 256
    encoder_depth: int = 3
    """Number of DownConv + CDAM + WSB scales (d = 3, Table VI)."""

    num_parameters_m: float = 16.71
    train_sequences: int = 280
    train_epochs: int = 200
    batch_size: int = 2
    crop_size: tuple[int, int] = (128, 128)
    learning_rate: float = 1e-4

    sequence_length: int = 40
    """L in Eq. (7) — temporal unroll length."""

    temporal_loss_start: int = 2
    """L0 in Eq. (7)."""

    temporal_loss_weight: float = 5.0
    """λ_TC."""

    occlusion_alpha: float = 50.0
    """α in occlusion mask M^k_{k-1} (Eq. 6)."""

    training_datasets: tuple[str, ...] = (
        "ECD",
        "HQF",
        "MVSEC",
        "high-speed HDR (640×480)",
        "CED (color)",
    )
