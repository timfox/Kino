"""Sagnac-assisted enhanced ϕ-OTDR DAS benchmark — Wang et al., arXiv:2606.05754."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SagnacPhiOtdrConfig:
    paper_arxiv: str = "arXiv:2606.05754"
    title: str = (
        "Sagnac-Assisted Enhanced ϕ-OTDR for Distributed Acoustic Sensing: "
        "A Standardized Benchmark and Engineering Evaluation Framework"
    )
    framework: str = "Sagnac-Phi-OTDR-DAS"

    sensing_distance_km: float = 10.0
    n_channels: int = 12
    n_classes: int = 6
    github_repo: str = "https://github.com/wawa-abc/das"

    # Balanced benchmark (Table 1)
    total_samples: int = 15419
    train_ratio: float = 0.8
    test_samples: int = 3084

    # Table 2 — Fusion CNN (best overall)
    fusion_accuracy: float = 89.79
    fusion_macro_f1: float = 89.83
    fusion_nar: float = 5.00
    fusion_fnr: float = 0.00
    fusion_latency_ms: float = 12.7901

    # Table 2 — baselines
    stft_svm_accuracy: float = 41.25
    stft_svm_macro_f1: float = 38.39
    mpe_zcr_svm_accuracy: float = 44.37
    psvm_accuracy: float = 56.46
    branch_b_cnn_accuracy: float = 86.04
    branch_b_cnn_macro_f1: float = 85.89

    # Table 4 — best channel grouping (search stage)
    best_grouping_accuracy: float = 78.75
    default_split_accuracy: float = 51.25  # front-half vs back-half

    event_classes: tuple[str, ...] = (
        "01_background",
        "02_dig",
        "03_knock",
        "04_water",
        "05_shake",
        "06_walk",
    )

    @property
    def train_samples(self) -> int:
        return int(round(self.total_samples * self.train_ratio))
