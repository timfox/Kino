"""Hyperparameters for gaze-head coordination cVAE (Liu et al., arXiv:2605.25810)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GazeHeadConfig:
    """MVA 2025 paper defaults (Sec. 2.3, 3.1)."""

    paper_arxiv: str = "arXiv:2605.25810"
    paper_doi: str = "10.23919/MVA65244.2025.11175115"
    conference: str = "MVA 2025, Kyoto"

    train_videos: int = 45_806
    test_videos: int = 676
    dataset: str = "CelebV-Text"
    source_fps: int = 25
    model_fps: int = 5

    seq_len: int = 12
    """~2.4 s window at 5 FPS."""

    latent_dim: int = 128
    hidden_dim: int = 256
    feature_dim: int = 8
    """Concatenated gaze + head + context features for encoder input."""

    train_steps: int = 60_000
    batch_size: int = 64
    learning_rate: float = 5e-5
    kl_weight: float = 1.0
    """λ in Eq. (1); annealed during training in full implementation."""

    num_generations_per_gaze: int = 30
    context_frames: int = 2
    """Past head poses c1, c2 for autoregressive long-term generation."""

    gaze_estimator: str = "Qin et al. appearance-based 3D gaze (CVPRW 2022)"
    video_synthesis: str = "ST-ED + ETH-XGaze (Sec. 2.3)"
