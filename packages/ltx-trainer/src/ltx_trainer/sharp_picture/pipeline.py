"""Sharp Picture framework card, bounds, and reference experiment tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sharp_picture.config import SharpPictureConfig
from ltx_trainer.sharp_picture.fourier import (
    degree_of_spectrum,
    fourier_expansion,
    random_positive_coefficients,
    sparsity_of_spectrum,
)
from ltx_trainer.sharp_picture.metrics import (
    cot_bound,
    gu_unperturbed,
    norm_proxy_l,
    onepass_bound,
    semi_analytic_bound,
)


def framework_card(cfg: SharpPictureConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SharpPictureConfig()
    return {
        "name": "SharpPicture",
        "paper": "arXiv:2605.20988",
        "title": "A Sharper Picture of Generalization in Transformers",
        "domain": "Boolean functions f: {0,1}^T → R via Fourier–Walsh spectrum",
        "learner": "Idealized low-sharpness interpolator (Definition 1)",
        "construction": "1.5-layer transformer, log-scaled position attention, no layer norm",
        "bound": "Oracle PAC-Bayes with domination by explicit construction Θ",
        "complexity": {
            "degree": "Df = max |A| with λ_A ≠ 0",
            "sparsity": "ω = number of non-zero Fourier coefficients",
            "sharpness": "Gu(ω, Df) ∈ O(ω D³_f)",
            "norm": "L(ω, Df, T) ∈ O(D³_f + log(T)² ω Df)",
        },
        "experiments": {
            "T": cfg.context_length,
            "m": cfg.training_samples_m,
            "sigma_sg": cfg.sigma_subgaussian,
            "degrees": list(cfg.degrees),
            "sparsities": list(cfg.sparsities),
        },
    }


def table_semi_analytic_bounds(
    m: int = 8192,
    sigma: float = 0.001,
    T: int = 20,
) -> dict[str, dict[str, float]]:
    """Semi-analytic bound grid (Fig. 1 left) for (Df, ω) at fixed m."""
    cfg = SharpPictureConfig(context_length=T, training_samples_m=m)
    out: dict[str, dict[str, float]] = {}
    for df in cfg.degrees:
        for w in cfg.sparsities:
            key = f"D{df}_w{w}"
            out[key] = {
                "degree": float(df),
                "sparsity": float(w),
                "gu": gu_unperturbed(w, df),
                "norm_l": norm_proxy_l(w, df, T),
                "bound": semi_analytic_bound(m, sigma, w, df, T, cfg.sigma_subgaussian, cfg.delta),
            }
    return out


def table_empirical_gap_reference() -> dict[str, str]:
    """Qualitative claims from Sec. 2.1 (Fig. 1 right)."""
    return {
        "m": "8192",
        "trend": "Generalization gap grows super-linearly in Df; slope increases with ω",
        "non_vacuous_example": "Df=2, ω=10, T=20: semi-analytic bound non-vacuous at m=8192",
        "analytic_p_loose": "Fully analytic P(σ) needs m≈2e9 vs semi-analytic at m=8192",
    }


def table_domination_assumption() -> dict[str, str]:
    """Sec. 2.2 — construction vs learned norm/sharpness."""
    return {
        "sharpness": "Explicit construction upper-bounds learned Tr(∇²L) by ~2 orders",
        "frobenius_norm": "Construction ‖Θ‖_F upper-bounds learned norm by ~2 orders",
        "pareto": "Construction not Pareto-optimal (Appendix D rescaling)",
    }


def parity_cot_comparison(
    T: int = 20,
    m: int = 8192,
    sigma: float = 0.001,
) -> dict[str, float]:
    """CoT vs one-pass bound scaling (Theorems 16–17)."""
    cfg = SharpPictureConfig()
    gu_cot = gu_unperturbed(cfg.cot_gu_omega, cfg.cot_gu_degree)
    gu_op = gu_unperturbed(cfg.onepass_gu_omega, cfg.onepass_gu_degree)
    l_cot = norm_proxy_l(cfg.cot_gu_omega, cfg.cot_gu_degree, T)
    l_op = norm_proxy_l(cfg.onepass_gu_omega, cfg.onepass_gu_degree, T)
    b_cot = cot_bound(T, m, sigma, cfg.sigma_subgaussian, gu_cot, l_cot)
    b_op = onepass_bound(T, m, sigma, cfg.sigma_subgaussian, gu_op, l_op)
    return {
        "T": float(T),
        "m": float(m),
        "sigma": sigma,
        "cot_bound": b_cot,
        "onepass_bound": b_op,
        "ratio_op_over_cot": b_op / max(b_cot, 1e-12),
        "scaling_note": "CoT error ~O(T); one-pass ~exp(T) under bound",
    }


def evaluation_demo() -> dict[str, Any]:
    """Smoke: Walsh target, Gu, and semi-analytic bound."""
    T = 20
    coeffs = random_positive_coefficients(3, 2, T)
    x = [1, 0, 1, 0, 1] + [0] * (T - 5)
    val = fourier_expansion(coeffs, x)
    return {
        "fourier_value": val,
        "degree": degree_of_spectrum(coeffs),
        "sparsity": sparsity_of_spectrum(coeffs),
        "gu_w10_d2": gu_unperturbed(10, 2),
        "bound_m8192_d2_w10": semi_analytic_bound(8192, 0.001, 10, 2, T),
        "parity_cot": parity_cot_comparison(T=T),
    }