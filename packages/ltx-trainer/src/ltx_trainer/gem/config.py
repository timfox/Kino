"""Configuration for GEM geometric entropy mixing (arXiv:2605.26121)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GemConfig:
    paper_arxiv: str = "2605.26121"
    paper_title: str = (
        "GEM: Geometric Entropy Mixing for Optimal LLM Data Curation"
    )
    author: str = "Yue Min et al. (Wizard Quant)"

    n_clusters: int = 24
    balance_lambda: float = 1.0
    max_mm_iters: int = 50
    e_step_mirror_steps: int = 5
    stop_tol: float = 1e-5
    eps: float = 1e-8

    # Teacher–student / GIS
    seed_fraction: float = 0.01
    gis_neighbors: int = 8
    gis_top_m: int = 5
    gis_beta: float = 0.5
    student_target_accuracy: float = 0.7513
    kmeans_baseline_accuracy: float = 0.7292

    # Pre-training experiment (Sec. 4)
    model_params_b: float = 1.1e9
    train_tokens_b: float = 25e9
    default_k_sensitivity: int = 36
