"""AdaMaG — Adaptive Manifold Guidance (arXiv:2605.20079)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdamagConfig:
    paper_arxiv: str = "2605.20079"
    paper_title: str = "Probability-Conserving Flow Guidance (AdaMaG)"

    # Default AdaMaG hyperparameters (paper Sec. 5)
    beta_default: float = 0.1
    gamma_default: float = 4.0
    omega_min_default: float = 1.0

    # Models evaluated
    models: tuple[str, ...] = ("SD3", "SD3.5", "Flux")
    baselines: tuple[str, ...] = ("CFG", "Rect-CFG++", "TAG", "APG")

    # Headline SD3 optimal-scale FID (Table 1, Ours row)
    sd3_fid_ours: float = 30.4
    sd3_fid_cfg: float = 32.4
    sd3_sat_ours: float = 0.51
    sd3_sat_cfg: float = 0.53

    # High-guidance SD3 ω=15
    sd3_high_guidance_omega: float = 15.0
    sd3_fid_ours_high: float = 35.6
    sd3_fid_cfg_high: float = 42.6

    solver_steps: int = 30
    eval_resolution: tuple[int, int] = (256, 256)
    eval_resolution_hr: tuple[int, int] = (1024, 1024)
