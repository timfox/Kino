"""Reference metrics (Huang et al., arXiv:2601.17271, Tables I–V)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cross360.config import PARAMS_M, PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table I — incomplete real-world ERP (M3D, S2D3D)
TABLE1_M3D_OURS: dict[str, Any] = {
    "dataset": "Matterport3D",
    "params_M": PARAMS_M,
    "AbsRel": 0.0955,
    "SqRel": 0.0706,
    "RMSE": 0.4163,
    "delta1": 90.78,
    "delta2": 97.11,
    "delta3": 98.88,
}

TABLE1_S2D3D_OURS: dict[str, Any] = {
    "dataset": "Stanford2D3D",
    "params_M": PARAMS_M,
    "AbsRel": 0.1211,
    "SqRel": 0.0740,
    "RMSE": 0.4042,
    "delta1": 85.66,
    "delta2": 97.09,
    "delta3": 99.13,
}

TABLE1_M3D_ELITE360D: dict[str, Any] = {
    "method": "Elite360D",
    "AbsRel": 0.1115,
    "RMSE": 0.4875,
}

# Table II — complete synthetic FOV
TABLE2_STRUCT3D_OURS: dict[str, Any] = {
    "dataset": "Structured3D",
    "AbsRel": 0.0361,
    "SqRel": 0.0127,
    "RMSE": 0.1030,
    "delta1": 97.88,
    "delta2": 99.25,
    "delta3": 99.63,
}

TABLE2_3D60_OURS: dict[str, Any] = {
    "dataset": "3D60",
    "AbsRel": 0.0526,
    "SqRel": 0.0177,
    "RMSE": 0.2121,
    "delta1": 97.40,
    "delta2": 99.55,
    "delta3": 99.86,
}

TABLE2_STRUCT3D_GLPANO: dict[str, Any] = {"method": "GLPanoDepth", "AbsRel": 0.0520, "RMSE": 0.1267}

# Table III — ablation on 3D60
TABLE3_ABLATION: list[dict[str, Any]] = [
    {"method": "Baseline", "AbsRel": 0.0656, "RMSE": 0.2445, "delta1": 95.49, "delta2": 99.17},
    {"method": "Baseline+CPFA", "AbsRel": 0.0533, "RMSE": 0.2172, "delta1": 97.07, "delta2": 99.43},
    {"method": "Baseline+PFAA", "AbsRel": 0.0576, "RMSE": 0.2230, "delta1": 96.84, "delta2": 99.42},
    {"method": "Ours (all)", "AbsRel": 0.0526, "RMSE": 0.2121, "delta1": 97.40, "delta2": 99.55},
]

# Table IV — TP patch count (3D60)
TABLE4_TP_PATCHES: list[dict[str, Any]] = [
    {"N": 10, "FoV_deg": 120, "FPS": 15.42, "GFLOPs": 172.12, "AbsRel": 0.0599, "RMSE": 0.2226, "delta1": 96.86},
    {"N": 18, "FoV_deg": 90, "FPS": 11.64, "GFLOPs": 172.74, "AbsRel": 0.0522, "RMSE": 0.2148, "delta1": 97.21},
    {"N": 26, "FoV_deg": 72, "FPS": 9.16, "GFLOPs": 173.36, "AbsRel": 0.0526, "RMSE": 0.2121, "delta1": 97.40},
    {"N": 46, "FoV_deg": 60, "FPS": 5.78, "GFLOPs": 174.90, "AbsRel": 0.0505, "RMSE": 0.2115, "delta1": 97.29},
]

# Table V — depth vs distance (3D60)
TABLE5_DISTANCE: list[dict[str, Any]] = [
    {"range_m": "[0,2)", "distribution_pct": 74.3, "AbsRel": 0.0559, "RMSE": 0.1279, "delta1": 97.84},
    {"range_m": "[2,4)", "distribution_pct": 18.4, "AbsRel": 0.0673, "RMSE": 0.3760, "delta1": 95.30},
    {"range_m": "[4,6)", "distribution_pct": 5.0, "AbsRel": 0.0788, "RMSE": 0.5392, "delta1": 96.14},
]


def table1_ours_m3d() -> dict[str, Any]:
    return dict(TABLE1_M3D_OURS)


def table2_ours_struct3d() -> dict[str, Any]:
    return dict(TABLE2_STRUCT3D_OURS)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "table1_m3d_ours": TABLE1_M3D_OURS,
        "table1_s2d3d_ours": TABLE1_S2D3D_OURS,
        "table2_struct3d_ours": TABLE2_STRUCT3D_OURS,
        "table2_3d60_ours": TABLE2_3D60_OURS,
        "table3_ablation": TABLE3_ABLATION,
        "table4_tp_patches": TABLE4_TP_PATCHES,
        "table5_distance": TABLE5_DISTANCE,
    }
