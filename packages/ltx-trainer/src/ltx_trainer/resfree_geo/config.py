"""Resolution-free neural geometric mapping surrogate (arXiv:2605.28551)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResfreeGeoConfig:
    paper_arxiv: str = "arXiv:2605.28551"
    authors: str = "Huang, Lui, Choi (CUHK)"
    architecture: str = "Multi-Resolution Enhanced U-Net"
    training_mode: str = "data-free (geometry-aware losses)"
    hardware_note: str = "NVIDIA RTX A6000 (paper)"
    problems: tuple[str, ...] = (
        "quasi-conformal (Beltrami)",
        "density-equalizing map (DEM)",
        "density-equalizing quasi-conformal (DEQ)",
    )
    baselines: tuple[str, ...] = (
        "FNO",
        "DeepONet",
        "RINO",
        "PhyGeoNet",
        "GINO",
        "LDEM",
        "LBS",
    )
    beltrami_train_epochs: int = 250_000
    deq_train_epochs: int = 12_000
    dem3d_train_epochs: int = 4_000
