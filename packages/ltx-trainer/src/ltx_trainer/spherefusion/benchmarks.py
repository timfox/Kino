"""Paper Table 1–2 (Yan et al., arXiv:2502.05859)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spherefusion.config import FPS_512, INFERENCE_SEC_512, PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — Stanford2D3D (S2D3D)
TABLE1_S2D3D = {
    "UniFuse": {"mre": 0.1114, "mae": 0.2082, "rmse": 0.3691, "rmse_log": 0.07213, "d1": 0.8711, "time_s": 0.02472},
    "SphereDepth": {"mre": 0.1158, "mae": 0.2323, "rmse": 0.4512, "rmse_log": 0.0754, "d1": 0.8666, "time_s": 0.0612},
    "PanoFormer": {"mre": None, "mae": None, "rmse": 0.30831, "rmse_log": None, "d1": 0.93941, "time_s": 0.1253},
    "SphereFusion": {
        "mre": 0.08991,
        "mae": 0.16541,
        "rmse": 0.31942,
        "rmse_log": 0.06111,
        "d1": 0.92572,
        "time_s": 0.01741,
    },
}

TABLE1_M3D = {
    "UniFuse": {"mre": 0.10632, "mae": 0.2814, "rmse": 0.4941, "time_s": 0.02472},
    "SphereDepth": {"mre": 0.1205, "mae": 0.3311, "rmse": 0.5922, "time_s": 0.0612},
    "SphereFusion": {"mre": 0.11453, "mae": 0.28522, "rmse": 0.48853, "rmse_log": 0.07332, "d1": 0.8701, "time_s": 0.01741},
}

TABLE1_360D = {
    "UniFuse": {"mre": 0.04663, "mae": 0.09962, "rmse": 0.1968, "time_s": 0.02212},
    "SphereDepth": {"mre": 0.0550, "mae": 0.1145, "rmse": 0.2364, "time_s": 0.05453},
    "SphereFusion": {
        "mre": 0.04171,
        "mae": 0.08941,
        "rmse": 0.1813,
        "rmse_log": 0.02862,
        "d1": 0.98692,
        "time_s": 0.01551,
    },
}

# Table 2 — 360D ablation
TABLE2_ABLATION = {
    "2d_only": {"mre": 0.0461, "rmse": 0.2081, "d1": 0.9833},
    "mesh_only": {"mre": 0.0572, "rmse": 0.2372, "d1": 0.9755},
    "BiFuse": {"mre": 0.0415, "rmse": 0.1824},
    "UniFuse": {"mre": 0.0427, "rmse": 0.1837},
    "GateFuse": {"mre": 0.0417, "rmse": 0.1813, "d1": 0.9869},
}


def beats_sphere_depth_s2d3d() -> bool:
    sf, sd = TABLE1_S2D3D["SphereFusion"], TABLE1_S2D3D["SphereDepth"]
    return sf["mre"] < sd["mre"] and sf["rmse"] < sd["rmse"]


def faster_than_unifuse_s2d3d() -> bool:
    return TABLE1_S2D3D["SphereFusion"]["time_s"] < TABLE1_S2D3D["UniFuse"]["time_s"]


def gatefuse_best_ablation() -> bool:
    return TABLE2_ABLATION["GateFuse"]["rmse"] <= TABLE2_ABLATION["BiFuse"]["rmse"]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "inference_sec_512": INFERENCE_SEC_512,
        "fps_512": FPS_512,
        "table1_s2d3d": TABLE1_S2D3D,
        "table1_m3d": TABLE1_M3D,
        "table1_360d": TABLE1_360D,
        "table2_ablation": TABLE2_ABLATION,
    }
