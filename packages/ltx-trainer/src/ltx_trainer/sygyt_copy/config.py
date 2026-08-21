"""Sygyt articulatory copy-synthesis — Cámara et al., arXiv:2606.04943 / DAFx26."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SygytCopyConfig:
    paper_arxiv: str = "arXiv:2606.04943"
    title: str = (
        "Differentiable Articulatory Copy-Synthesis of Biphonic Singing"
    )
    framework: str = "Sygyt-Copy"
    venue: str = "DAFx26"

    sample_rate_hz: int = 16000
    segments: int = 20
    singers: int = 5
    pitches: int = 10
    adam_iterations: int = 500
    adam_lr: float = 5e-3

    # Tract parameterization DOFs per frame (Sec. 3.3)
    articulator_dofs: int = 13
    bspline_dofs: int = 70
    bspline_control_points: int = 20

    # Table 2 — HFA dataset (LSD dB, lower better)
    hfa_artic_lsd: float = 13.84
    hfa_ddsp_lsd: float = 10.99
    hfa_bspline_lsd: float = 9.64

    # Table 2 — Bergevin dataset
    berg_artic_lsd: float = 14.53
    berg_ddsp_lsd: float = 10.71
    berg_bspline_lsd: float = 9.04

    # Relative LSD reduction vs articulator (Sec. 4.3)
    lsd_reduction_hfa_pct: float = 30.0
    lsd_reduction_berg_pct: float = 38.0

    # Overtone region (1–3 kHz)
    overtone_band_hz: tuple[int, int] = (1000, 3000)

    # Table 3 all-dataset overtone errors (B-spline)
    bspline_delta_er: float = 0.12
    bspline_delta_sot_db: float = 0.96
    bspline_delta_hpr: float = 5.48

    # Formant peak error Hz (Sec. 4.4)
    formant_peak_error_bspline_hz: float = 28.0
    formant_peak_error_artic_hz: float = 222.0
    formant_peak_error_ddsp_hz: float = 120.0

    # Table 4 ablation (full model)
    ablation_full_lsd: float = 9.34
    ablation_no_sublingual_lsd: float = 10.32
    ablation_no_damping_lsd: float = 9.48
    ablation_minimal_lsd: float = 10.45

    # Pitch detection (Sec. 4.2)
    diphonic_frame_pct: float = 99.0
    mean_overtone_enhancement_db: float = 37.6

    companion_url: str = "https://mateocamara.com/khoomei-supp-materials"
