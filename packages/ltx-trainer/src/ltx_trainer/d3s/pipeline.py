"""Framework card, demos, smoke for D3S Consensus (arXiv:2606.02906)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.d3s.config import D3SConfig
from ltx_trainer.d3s.paper_tables import (
    fig1b_commercial_stereo,
    fig5_prototype_metrics,
    knowledge_card,
    table_s2_simulation_comparison,
)
from ltx_trainer.d3s.simulation import confidence_threshold_ablation, full_pipeline_demo


def framework_card(cfg: D3SConfig | None = None) -> dict[str, Any]:
    cfg = cfg or D3SConfig()
    o = cfg.optics
    return {
        "name": "D3S Consensus",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "title": "Depth from Dual Differential Defocus and Stereo (D3S) Consensus",
        "task": "Physics-based sparse depth from dual-defocus binocular snapshot",
        "pipeline": [
            "Capture rectified pair (I0, I1) with stereo baseline + focus offset",
            "Search candidate depths Z_i and virtual baselines ΔX_j",
            "D3 depth Z_D3 via Eq. 10/17 on shifted pairs",
            "Consensus on Z and 1/Z (Eq. 13); confidence C_i (Eq. 14–16)",
            "Optional Marigold-DC densification",
        ],
        "components": [
            "d3_theory",
            "stereo_reprojection",
            "consensus_filter",
            "calibration_a_b",
        ],
        "prototype": {
            "baseline_mm": o.baseline_mm,
            "efl_mm": o.sensor_distance_mm,
            "virtual_baselines_mm": list(o.virtual_baselines_mm),
        },
        "working_range_m": list(cfg.working_range_m),
        "mae_target_cm": cfg.mae_target_cm,
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no raw Basler capture, rectification homography, or SIFT vanishing-point calibration.",
        "Eq. 17 uses supplement-style cubic a(ΔX), b(ΔX) anchors; not fitted on physical CUReT planes.",
        "Synthetic scenes approximate defocus; reported working range uses paper Fig. 5 anchors in smoke.",
        "Marigold-DC densification not bundled; domain shift noted in paper Sec. 5.5.",
        "Sparse output only — densification is optional downstream.",
    ]


def evaluation_demo(cfg: D3SConfig | None = None) -> dict[str, Any]:
    cfg = cfg or D3SConfig()
    demo = full_pipeline_demo(cfg=cfg)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "knowledge": knowledge_card(),
        "fig1b_stereo_products": fig1b_commercial_stereo(),
        "fig5_metrics": fig5_prototype_metrics(),
        "table_s2": table_s2_simulation_comparison(),
        "pipeline_demo": demo,
        "confidence_ablation": confidence_threshold_ablation(cfg=cfg),
    }


def evaluation_smoke(cfg: D3SConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    cfg = cfg or D3SConfig()
    fig5 = demo["fig5_metrics"]
    ours = next(r for r in demo["table_s2"] if "ours" in r["method"].lower())
    d435 = next(r for r in demo["fig1b_stereo_products"] if "D435" in r["name"])

    assert fig5["working_range_m_cthre_0_8"] == (0.3, 1.64)
    assert ours["baseline_mm"] < 5.0
    assert ours["efl_mm"] < 15.0
    assert ours["eps_d_px"] < 0.2
    assert ours["z_max_m"] > 1.5
    assert d435["significant_dim_mm"] > ours["efl_mm"] * 3

    ablation = demo["confidence_ablation"]
    assert ablation[-1]["confidence_threshold"] == 0.8
    assert ablation[-1]["density"] <= ablation[0]["density"]

    pipe = demo["pipeline_demo"]
    assert pipe["valid_fraction"] > 0.0
    assert np_finite(pipe["max_confidence"])

    return {
        "status": "ok",
        "paper": f"arXiv:{cfg.paper_arxiv}",
        "working_range_m": fig5["working_range_m_cthre_0_8"],
        "prototype_baseline_mm": cfg.optics.baseline_mm,
        "z_max_table_s2_m": ours["z_max_m"],
        "eps_d_at_zmax_px": ours["eps_d_px"],
        "demo_keys": list(demo.keys()),
    }


def np_finite(x: float) -> bool:
    import math

    return math.isfinite(x)
