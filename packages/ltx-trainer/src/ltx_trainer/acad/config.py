"""ACAD stub (arXiv:2605.22262)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AcadConfig:
    paper_arxiv: str = "arXiv:2605.22262"
    title: str = "Automatic contextual audio denoising"
    num_scene_classes: int = 6
    pairs_train_per_class: int = 10000
    pairs_val_test_per_class: int = 3000
    asc_test_accuracy_pct: float = 84.18
    # Table 2 headline: UNetTu-ASC vs UNet (SI-SDR)
    unet_si_sdr_db: float = 10.16
    unet_tu_asc_si_sdr_db: float = 12.12
    zenodo_dataset: str = "https://doi.org/10.5281/zenodo.20287453"
