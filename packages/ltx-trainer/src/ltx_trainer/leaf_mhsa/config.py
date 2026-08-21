"""Leaf MHSA — grapevine trait-to-spectra (SPIE 2025)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LeafMHSAConfig:
    doi: str = "10.1117/12.3061298"
    title: str = (
        "Leaf Spectral Reflectance Prediction Using Multi-Head Attention Neural Networks"
    )
    venue: str = "SPIE DCS 2025, Autonomous Air and Ground Sensing X, Vol. 13475, 134750V"

    n_traits: int = 16
    n_bands: int = 2101
    wavelength_nm_min: int = 400
    wavelength_nm_max: int = 2500
    band_step_nm: int = 1

    # Architecture (§2.3)
    embed_dim: int = 64
    n_heads: int = 8
    conv_filters: tuple[int, int] = (64, 32)
    conv_kernel: int = 20
    fc_hidden: int = 128
    dropout: float = 0.2

    # Training (§2.3)
    lr: float = 1e-3
    epochs: int = 300
    cv_folds: int = 5

    # Dataset (§2.1)
    n_leaves: int = 2305
    n_unique_samples: int = 992
    varieties: tuple[str, ...] = ("Sunpreme", "Flame Seedless", "Solbrio")
    collection_years: tuple[int, ...] = (2021, 2022, 2023)

    # Smoke / CPU demo (smaller spectrum for fast forward)
    demo_n_bands: int = 128
