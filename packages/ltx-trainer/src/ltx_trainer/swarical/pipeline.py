"""Swarical framework card, paper tables, and smoke demos (MM '24)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.swarical.config import (
    LocalizationMode,
    SwaricalConfig,
    camera_performance_table2,
    camera_specs_table1,
    fls_density_per_area,
    swarm_count,
)
from ltx_trainer.swarical.layout import LIMITATIONS
from ltx_trainer.swarical.localization import (
    PoseVector,
    intra_swarm_step,
    is_converged,
    localization_mode_summary,
)
from ltx_trainer.swarical.metrics import chamfer_distance, estimate_hd_from_camera_error, hausdorff_distance
from ltx_trainer.swarical.mock import grid_points, perturbed_grid
from ltx_trainer.swarical.planner import plan_from_points


def framework_card(cfg: SwaricalConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SwaricalConfig()
    d_min, d_max = fls_density_per_area(
        t_min=cfg.tracking_range_cm_min,
        t_max=cfg.tracking_range_cm_max,
        r=cfg.fls_sphere_radius_cm,
    )
    return {
        "name": "Swarical",
        "paper": cfg.paper_arxiv,
        "venue": "ACM MM '24",
        "doi": cfg.acm_doi,
        "companion_doi": cfg.companion_doi,
        "companion_arxiv": cfg.companion_arxiv,
        "github": cfg.github,
        "aruco_pose_repo": cfg.aruco_pose_repo,
        "idea": (
            "Integrated hierarchical localization for Flying Light Specks: offline planner (mesh→point cloud, "
            "k-Means swarms, FLS-trees + swarm-tree MST) plus online decentralized HC/ISR/RSF localization."
        ),
        "tracking": {
            "device": "Raspberry Camera Module 3 NoIR + ArUco markers",
            "range_cm": [cfg.tracking_range_cm_min, cfg.tracking_range_cm_max],
            "density_per_area": {"d_min": d_min, "d_max": d_max},
        },
        "planner": {
            "default_group_size_g": cfg.default_group_size_g,
            "swarm_count_formula": "n_G = ceil(F / G)",
            "trees": ["FLS-tree (intra-swarm)", "swarm-tree (inter-swarm)"],
        },
        "localization_modes": localization_mode_summary(),
        "recommended_mode": cfg.recommended_mode.value,
        "comparison_swarmer": {
            "speedup_at_least": cfg.swarmer_speedup_factor,
            "avg_distance_reduction_pct": cfg.swarmer_distance_reduction_pct,
        },
    }


def table_camera_specs() -> list[dict[str, Any]]:
    """Table 1 — Raspberry camera lenses."""
    return [
        {
            "lens": s.lens,
            "resolution_px": list(s.resolution_px),
            "fov_deg": list(s.fov_deg),
            "min_focus_mm": s.min_focus_mm,
            "weight_g": s.weight_g,
            "price_usd": s.price_usd,
        }
        for s in camera_specs_table1()
    ]


def table_camera_performance() -> list[dict[str, Any]]:
    """Table 2 — frame rate and processing latency."""
    return [
        {
            "resolution": r.resolution,
            "lens": r.lens,
            "fps": r.fps,
            "avg_camera_delay_ms": r.avg_camera_delay_ms,
            "avg_processing_ms": r.avg_processing_ms,
        }
        for r in camera_performance_table2()
    ]


def table_skateboard_planner() -> dict[str, Any]:
    """Sec. 5.2 — Skateboard point cloud and camera mix."""
    cfg = SwaricalConfig()
    top, side, bottom = cfg.skateboard_camera_mix_pct
    return {
        "shape": "Skateboard",
        "fls_illuminating": cfg.skateboard_fls_count,
        "standby_fls": cfg.skateboard_standby_fls,
        "camera_mix_pct": {"top": top, "side": side, "bottom": bottom},
        "tracking_range_cm": [cfg.tracking_range_cm_min, cfg.tracking_range_cm_max],
        "group_size_g": cfg.default_group_size_g,
        "n_swarms": swarm_count(cfg.skateboard_fls_count, cfg.default_group_size_g),
    }


def table_localization_skateboard_g50() -> dict[str, dict[str, float | str]]:
    """Fig. 13 qualitative ranking + HD plateau (Skateboard, G=50, 5° dead reckoning)."""
    cfg = SwaricalConfig()
    return {
        "ISR": {"rank": 1, "hd_plateau_mm_approx": cfg.skateboard_hd_plateau_mm, "notes": "Best HD/CD vs HC and RSF"},
        "HC": {"rank": 2, "hd_plateau_mm_approx": None, "notes": "Higher concurrency; worse than ISR"},
        "RSF": {"rank": 3, "hd_plateau_mm_approx": None, "notes": "Per-edge rounds; elevated HD/CD"},
    }


def table_swarmer_comparison() -> dict[str, Any]:
    """Sec. 5.4 — Swarical (ISR, G=50) vs SwarMer on Skateboard."""
    cfg = SwaricalConfig()
    return {
        "shape": "Skateboard",
        "swarical_mode": LocalizationMode.ISR.value,
        "group_size_g": cfg.default_group_size_g,
        "accuracy": "comparable Hausdorff distance",
        "time_to_localize": f">{cfg.swarmer_speedup_factor}x faster than SwarMer",
        "avg_distance_moved_reduction_pct": cfg.swarmer_distance_reduction_pct,
        "min_distance_moved_reduction_pct": 12.0,
        "message_overhead": "no SwarMer challenge phase (planner assigns anchors offline)",
    }


def table_group_size_sensitivity() -> dict[int, str]:
    """Fig. 14 — ISR HD/CD vs G (qualitative)."""
    return {
        5: "high HD/CD (deep swarm-tree, 43 swarms)",
        10: "high HD/CD (38 swarms)",
        50: "recommended balance (12 swarm-tree depth)",
        150: "lower HD/CD",
        200: "lower HD/CD (fewer swarms)",
    }


def planner_demo(cfg: SwaricalConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SwaricalConfig()
    pts = grid_points(rows=5, cols=5, spacing_cm=7.0)
    out = plan_from_points(pts, group_size_g=cfg.default_group_size_g, max_link_cm=cfg.tracking_range_cm_max)
    return {
        "n_fls": out.n_fls,
        "n_swarms": out.n_swarms,
        "fls_tree_edges": len(out.fls_tree_edges),
        "swarm_tree_edges": len(out.swarm_tree_edges),
        "dark_fls_inserted": out.dark_fls_inserted,
    }


def localization_demo() -> dict[str, Any]:
    ground = {0: PoseVector(0.0, 0.0, 0.0), 1: PoseVector(7.0, 0.0, 0.0)}
    est = {0: PoseVector(0.1, 0.0, 0.0), 1: PoseVector(6.8, 0.2, 0.0)}
    step = intra_swarm_step(ground, est)
    return {
        "correction_magnitude": step.magnitude(),
        "converged_at_0_5": is_converged(step, 0.5),
    }


def metrics_demo(cfg: SwaricalConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SwaricalConfig()
    planner_pts, est_pts = perturbed_grid(noise_cm=0.5)
    hd = hausdorff_distance(est_pts, planner_pts)
    cd = chamfer_distance(est_pts, planner_pts)
    analytical = estimate_hd_from_camera_error(planner_pts, pct_error=cfg.camera_pct_error_at_range)
    return {
        "hausdorff_cm": hd,
        "chamfer_cm_squared": cd,
        "analytical_hd_from_camera_pct_error_cm": analytical,
        "camera_pct_error": cfg.camera_pct_error_at_range,
    }


def evaluation_demo(cfg: SwaricalConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SwaricalConfig()
    from ltx_trainer.swarical.reproducibility import reproduction_walkthrough

    repro = reproduction_walkthrough()
    return {
        "framework": framework_card(cfg),
        "planner": planner_demo(cfg),
        "localization": localization_demo(),
        "metrics": metrics_demo(cfg),
        "limitations": list(LIMITATIONS),
        "reproducibility_companion": {
            "doi": cfg.companion_doi,
            "small_scale": repro["small_scale_demo"],
            "mode_ranking": repro["mode_comparison_4x4"]["ranking"],
        },
        "paper_tables": {
            "camera_specs": table_camera_specs(),
            "camera_performance": table_camera_performance(),
            "skateboard_planner": table_skateboard_planner(),
            "localization_g50": table_localization_skateboard_g50(),
            "swarmer_comparison": table_swarmer_comparison(),
            "group_size_sensitivity": table_group_size_sensitivity(),
        },
    }
