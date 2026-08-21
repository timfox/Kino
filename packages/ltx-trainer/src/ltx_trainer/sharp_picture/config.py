"""Configuration for Sharp Picture generalization (Lintilhac et al., arXiv:2605.20988)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SharpPictureConfig:
    """Defaults from paper experiments (Sec. 2, Appendix A)."""

    context_length: int = 20
    training_samples_m: int = 8192
    sigma_subgaussian: float = 0.01
    delta: float = 0.05
    # Typical Fourier targets in main experiments
    degrees: tuple[int, ...] = (1, 2, 3, 4, 5)
    sparsities: tuple[int, ...] = (1, 7, 14, 20)
    # Idealized learner (Definition 1)
    sharpness_weight_beta: float = 1.0
    norm_weight_alpha: float = 1.0
    # Gu(ω, Df) = a + b·ω·D³_f (Theorem 5)
    gu_coeff_a: float = 4.0
    gu_coeff_b: float = 32.0
    # Norm proxy L(ω, Df, T) — Theorem 10
    norm_scale_df: float = 16.0
    norm_scale_omega_df: float = 4.0
    norm_scale_log_t: float = 4.0
    # CoT parity (Sec. 3)
    cot_gu_degree: int = 2
    cot_gu_omega: int = 1
    onepass_gu_degree: int = 20
    onepass_gu_omega: int = 1
    # Empirical P(σ) placeholder when semi-analytic (Sec. 2.3)
    default_sigma_perturb: float = 0.001