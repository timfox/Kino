"""Channel-oriented EEG-to-music reconstruction (arXiv:2606.04040)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EegMusicConfig:
    paper_arxiv: str = "arXiv:2606.04040"
    title: str = "Channel-Oriented Design for EEG-to-Music Reconstruction"
    github: str = "https://github.com/jqin4749/EEG-to-Music"
    demo_page: str = "https://eegmusic626-design.github.io/eegmusic/"

    num_channels: int = 125
    sample_rate_hz: int = 125
    align_dim: int = 512
    num_heads: int = 16
    num_layers: int = 8
    channel_dropout: float = 0.2

    # Pretraining crops (Algorithm 1)
    global_views: int = 2
    local_views: int = 8
    global_crop_frac: tuple[float, float] = (0.5, 0.9)
    local_crop_frac: tuple[float, float] = (0.1, 0.5)

    # Table 1 — proposed (Ours)
    clap_score: float = 0.683
    id_50way: float = 0.487
    id_14way: float = 0.692
    genre_10way: float = 0.203

    # Strongest baseline (CBraMod)
    baseline_cbramod_50way: float = 0.402
    baseline_cbramod_clap: float = 0.641

    # EEG2Mel
    eeg2mel_50way: float = 0.259
    eeg2mel_clap: float = 0.588

    demo_time_samples: int = 125
    demo_batch: int = 2
