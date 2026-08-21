"""Paper anchors: Fig. 1b, Fig. 5, Table S2 (arXiv:2606.02906)."""

from __future__ import annotations

from typing import Any


def fig1b_commercial_stereo() -> list[dict[str, Any]]:
    """Working range (m) at MAE < 1 cm vs significant dimension (mm)."""
    return [
        {"name": "Intel RealSense D435", "significant_dim_mm": 50.0, "z_max_m": 1.2},
        {"name": "Intel RealSense D455", "significant_dim_mm": 95.0, "z_max_m": 6.0},
        {"name": "Stereolabs ZED 2", "significant_dim_mm": 120.0, "z_max_m": 5.0},
        {"name": "D3S prototype (ours)", "significant_dim_mm": 12.1, "z_max_m": 1.64, "z_min_m": 0.3},
    ]


def fig5_prototype_metrics() -> dict[str, Any]:
    """Sec. 5.3 — prototype accuracy and confidence trade-off."""
    return {
        "working_range_m_cthre_0_8": (0.3, 1.64),
        "mae_target_cm": 1.0,
        "confidence_threshold_default": 0.8,
        "real_world_visualization_cthre": 0.5,
        "density_drop_cthre_0_5_to_0_8": "sharp",
    }


def table_s2_simulation_comparison() -> list[dict[str, Any]]:
    """Supplement Table S2 — upper bound Z_max at MAE < 1 cm."""
    return [
        {"method": "Takeda et al. VISAPP '12", "baseline_mm": 14.0, "efl_mm": 30.0, "z_max_m": 0.37, "eps_d_px": 5.08},
        {"method": "Takeda et al. CVPR '13", "baseline_mm": 14.0, "efl_mm": 51.3, "z_max_m": 1.47, "eps_d_px": 0.54},
        {"method": "Tang et al. CVPR '17", "baseline_mm": 1.72, "efl_mm": 4.12, "z_max_m": 0.36, "eps_d_px": 0.36},
        {"method": "Liu et al. CVPR '25", "baseline_mm": 75.0, "efl_mm": 35.0, "z_max_m": 1.92, "eps_d_px": 1.21},
        {"method": "Ou et al. Vs. Comp. '25", "baseline_mm": 70.0, "efl_mm": 35.9, "z_max_m": 1.78, "eps_d_px": 1.35},
        {"method": "D3S Consensus (ours)", "baseline_mm": 3.84, "efl_mm": 12.1, "z_max_m": 1.56, "eps_d_px": 0.11},
    ]


def disparity_error_at_zmax(eps_d_px: float, z_max_m: float, *, baseline_mm: float, efl_mm: float, pitch_um: float = 2.0) -> float:
    """Eq. 18 sanity check — depth MAE from disparity EPE."""
    p = pitch_um * 1e-6
    s0 = efl_mm * 1e-3
    b = baseline_mm * 1e-3
    return eps_d_px * (z_max_m**2) * p / (s0 * b)


def knowledge_card() -> dict[str, Any]:
    return {
        "title": "D3S Consensus — Depth from Dual Differential Defocus and Stereo",
        "authors": "Luo, Xu, Chu, Alexander, Guo",
        "arxiv": "2606.02906",
        "prototype": {
            "baseline_mm": 3.84,
            "efl_mm": 12.1,
            "resolution": "902x1802 per view",
            "virtual_baselines_mm": [0.45, 0.50, 0.55],
        },
        "key_equations": ["D3 Eq. 10", "D3S consensus Eq. 13–16", "calibrated Eq. 17"],
        "optional_densify": "Marigold-DC (Viola et al., ICCV 2025)",
    }
