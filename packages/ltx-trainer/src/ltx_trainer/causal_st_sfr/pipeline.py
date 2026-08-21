"""Framework card, baselines, and paper excerpt tables for causal ST sound field reconstruction."""

from __future__ import annotations

from typing import Any

from ltx_trainer.causal_st_sfr.config import CausalStSfrConfig, DtuMeasuredCard, SimGeometryCard


def framework_card() -> dict[str, Any]:
    cfg = CausalStSfrConfig()
    return {
        "title": "Causal Spatio-Temporal Sound Field Reconstruction",
        "authors": "David Sundström, Filip Tronarp, Johan Lindström, Andreas Jakobsson (Lund University)",
        "paper": cfg.paper_url,
        "arxiv_id": cfg.paper_id,
        "problem": (
            "Estimate u(t_n, r̂_p) at P target locations from a causal length-W window of M microphones "
            "under additive Gaussian noise, without treating short-window DFT bins as independent."
        ),
        "model": {
            "wave_equation": "c⁻² ∂²u/∂t² − Δu = s(t,r)",
            "source_gp": "s ~ GP(0, Cs) with separable Cs = q w(r)w(r′) δ_Sa(r−r′) κ(t−t′)",
            "kappa": "band-limited flat spectrum → κ(Δ) = (sin ω₂Δ − sin ω₁Δ) / (Δ(ω₂−ω₁)), κ(0)=1",
            "covariance_integral": "Eq. (13)/(18) quadrature on sphere S_a",
            "far_field_limit": "Proposition 1 → classical diffuse-field sin((ω/c)d)/((ω/c)d)",
        },
        "estimator": {
            "name": "Spatio-temporal LMMSE / GP posterior mean",
            "equations": ["û = Kuy (Kyy + σ²I)⁻¹ y", "Σ_{u|y} = Kuu − Kuy (Kyy + σ²I)⁻¹ Kyu"],
            "complexity": "O((MW)³) precompute; O(PMW) online per sample",
        },
        "sample_selection": {
            "objective": "min_z tr(Σ_{u|ỹ}(z)) s.t. 1ᵀz = K",
            "relaxation": "capped simplex H_K + projected gradient on ϕ(z) = tr(Kuu) − tr((Kyy+σ²Z⁻²)⁻¹ KyuKuy)",
            "prune_greedy": "keep top ⌈ρK⌉ from relaxed z, then forward greedy on exact trace",
            "baselines": ["recent (newest lags first)", "uniform random"],
        },
        "contributions": [
            "Causal finite-window LMMSE with wave-equation-induced spatio-temporal covariance",
            "Far-field link to diffuse-field coherence while retaining finite-window temporal correlation",
            "Budget-constrained spatio-temporal sample selection over (mic, lag) pairs",
        ],
        "config_defaults": cfg.__dict__,
    }


def baselines_catalog() -> list[dict[str, str]]:
    return [
        {"id": "FD-KRR-Full", "role": "Offline full-record per-frequency diffuse KRR reference"},
        {"id": "FD-KRR", "role": "Causal length-W DFT; independent frequency bins (short W → leakage ignored)"},
        {"id": "FD-KRR (non-causal)", "role": "Centered window 2W−1 for causality ablation"},
        {"id": "FD-KRR-Trunc", "role": "High-res FD filter truncated to causal FIR (cf. spatial ANC [6])"},
        {"id": "Spatial", "role": "W=1 spatial-only covariance (zero temporal lag)"},
        {"id": "Spatio-temporal", "role": "Proposed joint causal ST LMMSE"},
    ]


def experiment_setup() -> dict[str, Any]:
    sim = SimGeometryCard()
    dtu = DtuMeasuredCard()
    cfg = CausalStSfrConfig()
    return {
        "excitation": "band-limited white Gaussian noise 70–1000 Hz",
        "simulated": {
            "rir": "image-source room 3×4×2.5 m, reflection 0.5, fs=8 kHz, c=343 m/s",
            "array": sim.mic_array,
            "reconstruction": sim.recon_disk,
            "monte_carlo": 50,
            "snr_db_default": 20,
            "sigma2_tuning": "20 log-spaced values in [1e-9, 1]; LOOM-CV on held-out mic",
        },
        "measured_dtu": {
            "obs": f"M={dtu.m_obs_draw} of {dtu.spherical_channels_total} spherical channels",
            "validation": f"P={dtu.p_linear_validation} linear array positions",
            "fs_hz": dtu.fs_downsample_hz,
            "causal_w": dtu.causal_horizon_w,
            "full_budget": dtu.full_st_budget_k,
        },
        "prior": {
            "q": cfg.q_source_intensity,
            "sphere_a_m": cfg.sphere_radius_a_m,
            "quadrature_q": cfg.n_quad_points_q,
            "band_hz": [cfg.f1_hz, cfg.f2_hz],
        },
        "selection_hyperparams": {
            "epsilon": cfg.relax_eps,
            "rho": cfg.prune_rho,
            "pg_iters": cfg.n_projected_grad_iters,
        },
    }


def figure_excerpts() -> dict[str, Any]:
    """Qualitative headline numbers quoted in paper §VI (not re-simulated here)."""
    return {
        "fig2_diffuse_window_sweep": {
            "note": "Causal spatio-temporal approaches FD-KRR-Full near W≈5; FD-KRR needs much longer W",
            "window_w_sweep": [2, 100],
        },
        "fig4_snr_sweep_db": {
            "causal_w": 10,
            "snr_db": [-5, 35],
            "note": "Low SNR widens gap to full-record FD reference; high SNR methods converge",
        },
        "fig5_source_radius_m": {"sweep_log": [0.1, 10.0], "note": "NMSE insensitive to assumed sphere radius a"},
        "fig6_quadrature_q": {"sweep": [1, 100], "stable_above": 100, "production_q": 1000},
        "fig7_measured_window": {"w_sweep": [2, 20], "note": "ST causal beats finite-window FD baselines; gap to FD-KRR-Full at W=20"},
        "fig11_sampling_budget": {
            "k_sweep": [50, 1000],
            "headline": "≈half the spatio-temporal samples for similar NMSE vs random/recent",
            "cost_scaling": "factorization ∝ α³, memory α², online α for K=αMW",
        },
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "baselines": baselines_catalog(),
        "experiments": experiment_setup(),
        "figures": figure_excerpts(),
        "appendix_c": {
            "finite_window_coupling": (
                "Rectangular causal window couples DFT bins via Dirichlet kernel; "
                "independent-bin FD-KRR is suboptimal for small W (Fig. 1)."
            ),
        },
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.causal_st_sfr.mock import evaluation_smoke

    return {"demo": evaluation_smoke()}
