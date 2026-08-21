"""cSTMM framework card and paper tables (arXiv:2605.25512)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cstmm.config import CstmmConfig
from ltx_trainer.cstmm.density import limiting_case_label
from ltx_trainer.cstmm.geometry import normalize_observation
from ltx_trainer.cstmm.layout import LIMITATIONS
from ltx_trainer.cstmm.mm import (
    build_canonical_a,
    hca_concentration,
    hca_eigenvalues,
    phi_auxiliary,
    scatter_matrix,
    update_weights,
)
from ltx_trainer.cstmm.mock import random_unit_vectors, toy_mm_step


def framework_card(cfg: CstmmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CstmmConfig()
    return {
        "name": "cSTMM",
        "paper": cfg.paper_arxiv,
        "author": cfg.author,
        "idea": (
            "Complex spherical Student's t mixture model for mask-based blind speech "
            "separation. Unifies cACGMM (ν=M), cBMM (ν→∞), and cWMM (rank-one, ν→∞) "
            "via degrees-of-freedom ν on the complex unit sphere."
        ),
        "observation": "z_tf = y_tf / ||y_tf||_2 (Eq. 1)",
        "density": "p(z|A,ν) ∝ (1 - 2/ν z^H A z)^{-(ν+M)/2} (Eq. 8)",
        "estimation": "Generalized MM: responsibilities (4), ϕ (14), weights (15), HCA eigenvalues (24)",
        "nu_star": cfg.nu_star,
        "stft": {"fft": cfg.stft_fft, "hop": cfg.stft_hop, "sr_hz": cfg.sample_rate_hz},
        "data": "LibriSpeech + MIRD RIRs, 18 acoustic conditions",
        "defaults": cfg.__dict__,
    }


def table_i_sdri() -> list[dict[str, Any]]:
    """Table 1 — test-set SDRi (dB), ν* = 1 vs ν = M."""
    return [
        {"M": 2, "N": 3, "rt60_ms": 160, "nu_star": 10.640, "nu_M": 10.580, "delta": 0.060, "se": 0.021, "p_holm": 6.4e-8, "dz": 0.18},
        {"M": 2, "N": 3, "rt60_ms": 360, "nu_star": 8.815, "nu_M": 8.790, "delta": 0.025, "se": 0.027, "p_holm": 6.4e-6, "dz": 0.06},
        {"M": 2, "N": 3, "rt60_ms": 610, "nu_star": 6.259, "nu_M": 6.210, "delta": 0.049, "se": 0.042, "p_holm": 0.001, "dz": 0.07},
        {"M": 3, "N": 2, "rt60_ms": 160, "nu_star": 14.092, "nu_M": 13.835, "delta": 0.258, "se": 0.017, "p_holm": 5.5e-34, "dz": 0.96},
        {"M": 3, "N": 2, "rt60_ms": 360, "nu_star": 13.466, "nu_M": 13.293, "delta": 0.172, "se": 0.010, "p_holm": 1.7e-37, "dz": 1.09},
        {"M": 3, "N": 2, "rt60_ms": 610, "nu_star": 11.796, "nu_M": 11.659, "delta": 0.137, "se": 0.013, "p_holm": 2.1e-29, "dz": 0.65},
        {"M": 3, "N": 3, "rt60_ms": 160, "nu_star": 12.530, "nu_M": 12.431, "delta": 0.099, "se": 0.022, "p_holm": 4.0e-9, "dz": 0.28},
        {"M": 3, "N": 3, "rt60_ms": 360, "nu_star": 11.422, "nu_M": 11.237, "delta": 0.185, "se": 0.021, "p_holm": 7.7e-18, "dz": 0.56},
        {"M": 3, "N": 3, "rt60_ms": 610, "nu_star": 9.268, "nu_M": 9.064, "delta": 0.204, "se": 0.040, "p_holm": 3.6e-13, "dz": 0.32},
        {"M": 4, "N": 2, "rt60_ms": 160, "nu_star": 13.920, "nu_M": 13.544, "delta": 0.377, "se": 0.022, "p_holm": 4.1e-36, "dz": 1.06},
        {"M": 4, "N": 2, "rt60_ms": 360, "nu_star": 13.578, "nu_M": 13.346, "delta": 0.233, "se": 0.013, "p_holm": 1.8e-35, "dz": 1.13},
        {"M": 4, "N": 2, "rt60_ms": 610, "nu_star": 12.584, "nu_M": 12.393, "delta": 0.191, "se": 0.016, "p_holm": 2.7e-28, "dz": 0.74},
        {"M": 4, "N": 3, "rt60_ms": 160, "nu_star": 12.611, "nu_M": 12.258, "delta": 0.353, "se": 0.026, "p_holm": 3.2e-30, "dz": 0.85},
        {"M": 4, "N": 3, "rt60_ms": 360, "nu_star": 12.470, "nu_M": 12.119, "delta": 0.351, "se": 0.030, "p_holm": 2.1e-28, "dz": 0.73},
        {"M": 4, "N": 3, "rt60_ms": 610, "nu_star": 11.039, "nu_M": 10.506, "delta": 0.533, "se": 0.032, "p_holm": 4.0e-35, "dz": 1.05},
        {"M": 4, "N": 4, "rt60_ms": 160, "nu_star": 13.041, "nu_M": 12.641, "delta": 0.400, "se": 0.022, "p_holm": 4.0e-35, "dz": 1.13},
        {"M": 4, "N": 4, "rt60_ms": 360, "nu_star": 12.241, "nu_M": 11.881, "delta": 0.359, "se": 0.029, "p_holm": 1.3e-25, "dz": 0.78},
        {"M": 4, "N": 4, "rt60_ms": 610, "nu_star": 10.658, "nu_M": 10.140, "delta": 0.518, "se": 0.044, "p_holm": 1.1e-27, "dz": 0.74},
    ]


def figure_i_model_recovery() -> list[dict[str, Any]]:
    """Fig. 1 — mean |ΔSDRi| when cSTMM recovers included models (RT60=610 ms)."""
    return [
        {"included_model": "cACGMM", "nu_setting": "ν = M", "mean_abs_sdri_diff_db": 1.7e-13},
        {"included_model": "cBMM", "nu_setting": "ν = 10⁴", "mean_abs_sdri_diff_db": 2.8e-3},
        {"included_model": "cWMM", "nu_setting": "rank-one, ν = 10⁴", "mean_abs_sdri_diff_db": 7.6e-4},
    ]


def headline_results() -> dict[str, Any]:
    rows = table_i_sdri()
    deltas = [r["delta"] for r in rows]
    return {
        "nu_star": 1.0,
        "comparison": "cSTMM ν* = 1 vs cACGMM-equivalent ν = M",
        "mean_delta_db": float(np.mean(deltas)),
        "delta_range_db": (float(min(deltas)), float(max(deltas))),
        "all_conditions_positive": all(d > 0 for d in deltas),
        "holm_significant_0_05": True,
        "model_recovery": figure_i_model_recovery(),
    }


def evaluation_demo(cfg: CstmmConfig | None = None) -> dict[str, Any]:
    """Toy sphere observations + one MM/HCA step (no audio)."""
    cfg = cfg or CstmmConfig()
    rng = np.random.default_rng(7)
    m, n_src = 3, 2
    z_list = random_unit_vectors(40, m, rng)
    matrices = [build_canonical_a(np.eye(m, dtype=np.complex128), np.array([0, -0.1, -0.05])),
                build_canonical_a(np.eye(m, dtype=np.complex128), np.array([0, -0.05, -0.1]))]
    weights = np.array([0.5, 0.5])
    nu = cfg.nu_star
    gamma, w_new = toy_mm_step(z_list, weights, matrices, nu, m)
    phi = phi_auxiliary(z_list, matrices, gamma, nu)
    s0 = scatter_matrix(z_list, gamma, phi, 0, nu)
    hca_lam = hca_eigenvalues(s0, float(gamma[:, 0].sum()), nu)
    hca_kappa = hca_concentration(s0, float(gamma[:, 0].sum()), m)

    y_demo = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    z_demo = normalize_observation(y_demo)

    return {
        "m_channels": m,
        "n_sources": n_src,
        "nu": nu,
        "limiting_case_nu_star": limiting_case_label(nu, m),
        "limiting_case_nu_M": limiting_case_label(float(m), m),
        "limiting_case_nu_large": limiting_case_label(cfg.nu_large_limit, m),
        "weights_updated": w_new.tolist(),
        "gamma_shape": list(gamma.shape),
        "hca_eigenvalues_tail": hca_lam[1:].tolist(),
        "hca_kappa": hca_kappa,
        "normalized_observation_norm": float(np.linalg.norm(z_demo)) if z_demo is not None else None,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_sdri": table_i_sdri(),
        "figure_i_model_recovery": figure_i_model_recovery(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
