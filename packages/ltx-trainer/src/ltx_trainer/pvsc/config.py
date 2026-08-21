"""PVSC: perception-aware video semantic communication (arXiv:2605.19397)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PVSCConfig:
    paper_arxiv: str = "arXiv:2605.19397"
    venue_note: str = "IEEE ICC Workshops 2026 (partial)"
    architecture_note: str = "DC blocks + ViT/contextual ViT feature coding (no explicit MV branch)"
    feature_channels: int = 128
    hyper_channels: int = 128
    gop_length: int = 7
    eta_spectral: float = 0.2
    lambda_rec: float = 1.0
    lambda_per: float = 0.8
    lambda_adv: float = 0.1
    train_snr_db: float = 10.0
    eval_snr_db: float = 6.0
    rate_set_size: int = 16
    params_m: float = 35.4
