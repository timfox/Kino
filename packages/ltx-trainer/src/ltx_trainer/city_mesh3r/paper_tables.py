"""Paper tables for City-Mesh3R (arXiv:2605.30310)."""

from __future__ import annotations

from typing import Any


def table_i_gauu_scene() -> list[dict[str, Any]]:
    """Table 1 — GauU-Scene surface P/R/F1 and runtime (minutes)."""
    return [
        {
            "scene": "CUHK-LOWER",
            "CityGS-v2": {"P": 0.1312, "R": 0.0820, "F1": 0.1009, "time_min": 340.90},
            "CityGS-X": {"P": 0.1327, "R": 0.0758, "F1": 0.0965, "time_min": 75.00},
            "Ours": {"P": 0.1420, "R": 0.0910, "F1": 0.1110, "time_min": 95.00},
        },
        {
            "scene": "CUHK-UPPER",
            "CityGS-v2": {"P": 0.2203, "R": 0.0953, "F1": 0.1330, "time_min": 371.05},
            "CityGS-X": {"P": 0.2950, "R": 0.1510, "F1": 0.2000, "time_min": 74.00},
            "Ours": {"P": 0.2680, "R": 0.1120, "F1": 0.1580, "time_min": 113.00},
        },
        {
            "scene": "LFLS",
            "CityGS-v2": {"P": 0.0901, "R": 0.0666, "F1": 0.0791, "time_min": 296.90},
            "CityGS-X": {"P": 0.0978, "R": 0.0599, "F1": 0.0743, "time_min": 87.00},
            "Ours": {"P": 0.0951, "R": 0.0777, "F1": 0.0855, "time_min": 83.00},
        },
        {
            "scene": "SZIIT",
            "CityGS-v2": {"P": 0.1513, "R": 0.0488, "F1": 0.0738, "time_min": 299.07},
            "CityGS-X": {"P": 0.1849, "R": 0.0538, "F1": 0.0833, "time_min": 94.00},
            "Ours": {"P": 0.1925, "R": 0.0647, "F1": 0.0968, "time_min": 92.00},
        },
    ]


def table_ii_sfm_efficiency() -> list[dict[str, Any]]:
    """Table 2 — clustering SfM vs MASt3R-COLMAP / GLOMAP."""
    return [
        {"method": "MASt3R-COLMAP (all images)", "time_hrs": 53.88, "reproj_err_px": 1.17},
        {"method": "MASt3R-GLOMAP (all images)", "time_hrs": 19.45, "reproj_err_px": 1.856},
        {"method": "Ours (Clustering + MASt3R-COLMAP)", "time_hrs": 2.74, "reproj_err_px": 1.38},
    ]


def table_iii_clustering_ablation() -> list[dict[str, Any]]:
    """Table 3 — SLPA vs N-cut vs agglomerative."""
    return [
        {"method": "Our Clustering (SLPA)", "reproj_err": 1.38, "pct_img_rejected": 4.373},
        {"method": "N-cut Clustering", "reproj_err": 1.441, "pct_img_rejected": 9.78},
        {"method": "Agglomerative Clustering", "reproj_err": 1.563, "pct_img_rejected": 10.27},
    ]


def table_iv_poisson_ablation() -> list[dict[str, Any]]:
    """Table 4 — Poisson only vs full refinement."""
    return [
        {"scene": "CUHK-LOWER", "Poisson Only": {"P": 0.131, "R": 0.049, "F1": 0.071},
         "Poisson + Opt/Remesh": {"P": 0.142, "R": 0.091, "F1": 0.111}},
        {"scene": "SZIIT", "Poisson Only": {"P": 0.1321, "R": 0.0163, "F1": 0.029},
         "Poisson + Opt/Remesh": {"P": 0.1925, "R": 0.0647, "F1": 0.0968}},
    ]


def table_v_conremesh_ablation() -> list[dict[str, Any]]:
    """Table 5 — Ours vs Continuous-Remeshing under fixed vertex budget."""
    return [
        {"scene": "CUHK-LOWER", "Ours": {"P": 0.142, "R": 0.091, "F1": 0.111},
         "ConRemesh": {"P": 0.1350, "R": 0.0632, "F1": 0.0861}},
        {"scene": "SZIIT", "Ours": {"P": 0.1925, "R": 0.0647, "F1": 0.0968},
         "ConRemesh": {"P": 0.1858, "R": 0.0317, "F1": 0.0542}},
    ]


def table_vi_garden_mesh_quality() -> dict[str, dict[str, float]]:
    """Table 6 — MipNeRF-360 Garden mesh quality."""
    return {
        "Ours": {"AR": 3.025, "ANG": 1.92, "DTR": 0.0, "NME": 0.0039, "NMV": 0.0027, "VVD": 0.505, "CC": 9, "IBL": 24},
        "MiLo": {"AR": 15.652, "ANG": 28.83, "DTR": 0.0003, "NME": 0.0, "NMV": 0.0014, "VVD": 1.421, "CC": 918, "IBL": 61},
        "MeshSplatting": {"AR": 5.224, "ANG": 15.25, "DTR": 0.0, "NME": 27.34, "NMV": 31.34, "VVD": 3.045, "CC": 340, "IBL": 6554},
        "RadianceMesh": {"AR": 3.147, "ANG": 2.93, "DTR": 0.0, "NME": 1.28, "NMV": 6.26, "VVD": 1.914, "CC": 232, "IBL": 58},
    }
