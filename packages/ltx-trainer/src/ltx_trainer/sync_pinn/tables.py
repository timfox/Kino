"""Paper figure/table excerpts for benchmarks (arXiv:2601.00178)."""

from __future__ import annotations

from typing import Any


def fig2_baseline_setting() -> dict[str, Any]:
    return {
        "N": 10,
        "K": 0.05,
        "topology": "fully_connected",
        "omega_uniform": "[-π, π]",
        "theta0_uniform": "[-π/2, π/2]",
        "R_star": 1.0,
        "t_star": 2.0,
        "qualitative": "Uncontrolled R stays low; PINN control reaches R≈1 before t* and persists.",
    }


def fig4_intrinsic_cost_trends() -> dict[str, Any]:
    """Integrated cost E vs heterogeneity δ and coupling K (qualitative trends)."""
    return {
        "E_vs_delta_heterogeneity": "monotone increase with frequency spread δ",
        "E_vs_K_coupling": "monotone decrease with coupling K",
        "runs_per_point": 20,
        "note": "δ and K oppose each other as intrinsic sync difficulty knobs.",
    }


def fig5_target_cost_trends() -> dict[str, Any]:
    return {
        "E_vs_R_star": "monotone increase; sharp rise as R*→1",
        "E_vs_t_star": "monotone decrease; faster targets cost more",
        "runs_per_point": 20,
    }


def fig6_baseline_comparison_kuramoto() -> dict[str, Any]:
    return {
        "phase_linear_k_theta": 1.5,
        "phase_nonlinear_k_theta": 1.5,
        "frequency_compensation_k_omega": 1.0,
        "qualitative_P_t": "Phase feedback shows early transient peaks; PINN smooth.",
        "qualitative_E": "PINN cumulative cost comparable to frequency compensation.",
    }


def fig7_sakaguchi_frustrated() -> dict[str, Any]:
    return {
        "alpha": "π/2",
        "K": 0.4,
        "frequency_compensation": "fails to synchronize (no analytic reference)",
        "phase_feedback": "synchronizes with large early P(t) peaks",
        "pinn": "smooth control, lower E than phase feedback",
    }


def fig8_noise_robustness() -> dict[str, Any]:
    return {
        "sigma_range": "0 to moderate (fixed K, R*=1)",
        "R_mean_vs_sigma": "gradual decrease, remains near unity for moderate σ",
        "ER_vs_sigma": "monotone increase",
        "E_vs_sigma": "smooth increase (offline control fixed)",
    }


def headline_results() -> dict[str, Any]:
    return {
        "method": "Physics-informed neural control of Kuramoto synchronization",
        "targets": "Prescribe R* and t*; enforce persistence R(t)≥R* for all t≥t*",
        "vs_phase_feedback": "Avoids impulsive early transients in P(t)",
        "vs_freq_compensation": "Matches low E in gradient Kuramoto; still works when α≠0",
        "noise": "Robust ⟨R⟩ with modest ER growth under fixed offline u(t)",
    }
