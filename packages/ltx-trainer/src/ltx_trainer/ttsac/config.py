"""TT-SAC configuration (arXiv:2605.25488)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TTSACConfig:
    """Reference hyperparameters from the TT-SAC paper."""

    paper_arxiv: str = "arXiv:2605.25488"
    default_K: int = 3
    fps: int = 25
    audio_hz: int = 16000
    crop_size: int = 512
    datasets: tuple[str, ...] = ("Hallo", "CelebV-HQ", "RAVDESS")
    generators: tuple[str, ...] = (
        "SadTalker",
        "Sonic",
        "AniTalker",
        "FLOAT",
        "JoyVASA",
    )
    k_candidates: tuple[int, ...] = (0, 1, 2, 3, 5, 10)
    feature_dim: int = 32
