"""NeR-SC: neural representation for screen content video (arXiv:2605.27024)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NeRSCConfig:
    paper_arxiv: str = "arXiv:2605.27024"
    title: str = "NeR-SC: Adapting Neural Video Representation to Screen Content"
    backbone: str = "SNeRV"
    crop_resolution: tuple[int, int] = (960, 1920)  # H, W per HNeRV [6]

    # Architecture (Sec. III).
    encoder_stages: int = 5
    default_params_m: float = 3.0
    palette_size_k: int = 64
    palette_temperature: float = 1.0
    mgf_stages: tuple[str, ...] = ("F3", "F4", "F5")
    mgf_fuse_channels: int = 64

    # Training (Sec. IV-B).
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    batch_size: int = 8
    epochs: int = 300
    loss: str = "L2"

    # Embedding-level skip (Sec. III-D).
    skip_threshold_tau: float = 0.005
    skip_fps_gain: float = 20.7
    decode_fps_with_skip: float = 61.7
    decode_fps_baseline: float = 41.0

    # Headline results (Sec. V).
    psnr_dscvc_avg_db: float = 40.32
    psnr_vcd_avg_db: float = 41.73
    msssim_dscvc_avg: float = 0.9942
    msssim_vcd_avg: float = 0.9919
    snerv_margin_dscvc_db: float = 0.54
    snerv_margin_vcd_db: float = 0.11

    # Datasets.
    dscvc_test_videos: tuple[str, ...] = tuple(f"Video {i}" for i in range(11, 21))
    vcd_sequences: tuple[str, ...] = (
        "th001",
        "th002",
        "thbb001",
        "thbb002",
        "thob001",
        "thob002",
    )
