"""Sparse 4D bootstrapped cross-validation constants (arXiv:2605.19160)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sparse4dBcvConfig:
    paper_arxiv: str = "2605.19160"
    paper_title: str = (
        "An evaluation framework for sparse 4D (3D + time) imaging reconstruction "
        "via bootstrapped cross-validation"
    )

    example_reconstructor: str = "4D-ONIX"
    dataset_name: str = "water_droplet_collision"
    dataset_doi: str = "10.6084/m9.figshare.28533098"

    num_experiments: int = 16
    num_time_steps: int = 75
    volume_shape: tuple[int, int, int] = (128, 128, 128)
    num_projections_full: int = 16
    projection_span_deg: float = 180.0

    sparse_projection_counts: tuple[int, ...] = (2, 4, 8)
    ultrasparse_projections_per_experiment: int = 4
    ultrasparse_experiment_counts: tuple[int, ...] = (1, 2, 4, 8)

    num_bootstrap_subsets: int = 100
    num_cv_pairs: int = 1000

    metrics: tuple[str, ...] = ("MSE", "PSNR", "DSSIM", "NMI", "NCC", "FHC")

    institutions: tuple[str, ...] = (
        "Lund University (Synchrotron Radiation Research / NanoLund)",
        "University College London",
    )
