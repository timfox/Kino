"""ASAP / Anatomical DPO / HAF-Bench (Li et al., arXiv:2605.25759)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ASAPConfig:
    """Defaults from paper Sec. 6.1 (FLUX backbone)."""

    spatial_weight_alpha: float = 10.0
    """α in Eq. (4) localized velocity loss."""

    margin_tau: float = 0.01
    """τ in Eq. (6) margin-bounded regression."""

    dpo_beta: float = 5000.0
    """β in Eq. (3) sigmoid DPO (Diffusion-DPO scale)."""

    # HAP pair quality gates (Sec. 4.1, dataset stats)
    region_ssim_max: float = 0.65
    region_lpips_min: float = 0.25
    background_ssim_min: float = 0.95
    background_lpips_max: float = 0.02


@dataclass
class HAFBenchConfig:
    """HAF-Bench taxonomy (Sec. 5)."""

    prompts_per_category: int = 100
    n_categories: int = 5
