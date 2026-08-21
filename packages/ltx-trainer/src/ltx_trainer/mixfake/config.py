"""Configuration for MixFake (arXiv:2605.23201)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MixFakeConfig:
    paper_arxiv: str = "arXiv:2605.23201"
    github: str = "https://github.com/saltfish233/MixFake"
    sample_rate_hz: int = 16000
    clip_seconds: float = 4.0
    snr_levels_db: tuple[int, ...] = (-5, 0, 5, 10, 15, 20)
    mix_ratio: int = 4
    backbone: str = "XLSR-AASIST"
    ssl_encoder: str = "XLS-R"
    learning_rate: float = 5e-3
    batch_size: int = 32
    epochs: int = 30


@dataclass
class MixFakeDatasetStats:
    total_samples: int = 252_500
    total_hours: float = 673.69
    single_source_hours: float = 510.59
    mixed_source_hours: float = 163.10
    train_samples: int = 116_000
    dev_samples: int = 20_500
    eval_samples: int = 116_000
    speech_algorithms: int = 19
    music_algorithms: int = 10
    env_algorithms: int = 3
