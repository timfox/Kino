"""Paper tables (Yan et al., arXiv:2503.06129 / ACM TOMM)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vuboiqa.config import FLOPS_G, PARAMS_M, PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — four OIQA databases
TABLE1_OIQA_MAIN = {
    "S-PSNR": {"cviq_srcc": 0.708, "oiqa_srcc": 0.599, "jufe_srcc": 0.355, "oiq_srcc": 0.302},
    "WS-PSNR": {"cviq_srcc": 0.672, "oiqa_srcc": 0.581, "jufe_srcc": 0.353, "oiq_srcc": 0.295},
    "Assessor360": {"cviq_srcc": 0.976, "oiqa_srcc": 0.974, "jufe_srcc": 0.694, "oiq_srcc": 0.790, "params_m": 88.2, "gflops": 230.5},
    "MC360IQA": {"cviq_srcc": 0.950, "oiqa_srcc": 0.924, "jufe_srcc": 0.620, "oiq_srcc": 0.721},
    "VGCN": {"cviq_srcc": 0.965, "oiqa_srcc": 0.958, "jufe_srcc": 0.464, "oiq_srcc": 0.705},
    "VU-BOIQA": {
        "cviq_srcc": 0.963,
        "cviq_plcc": 0.958,
        "oiqa_srcc": 0.976,
        "oiqa_plcc": 0.973,
        "jufe_srcc": 0.782,
        "jufe_plcc": 0.781,
        "oiq_srcc": 0.766,
        "oiq_plcc": 0.749,
        "params_m": PARAMS_M,
        "gflops": FLOPS_G,
    },
}

# Table 2 — cross-database
TABLE2_CROSS_DB = {
    "MC360IQA": {"oiq_plcc": 0.290, "oiq_srcc": 0.278, "jufe_rmse": 1.063},
    "VGCN": {"oiq_plcc": 0.426, "oiq_srcc": 0.418, "jufe_rmse": 0.415},
    "Assessor360": {"oiq_plcc": 0.357, "oiq_srcc": 0.367, "jufe_rmse": 0.428},
    "VU-BOIQA": {"oiq_plcc": 0.458, "oiq_srcc": 0.472, "jufe_rmse": 0.125},
}

# Table 3 — number of patches
TABLE3_PATCH_COUNT = {
    5: {"jufe_plcc": 0.731, "jufe_srcc": 0.726, "oiq_plcc": 0.747, "oiq_srcc": 0.727},
    10: {"jufe_plcc": 0.782, "jufe_srcc": 0.781, "oiq_plcc": 0.766, "oiq_srcc": 0.749},
    15: {"jufe_plcc": 0.782, "jufe_srcc": 0.782, "oiq_plcc": 0.783, "oiq_srcc": 0.762},
    20: {"jufe_plcc": 0.503, "jufe_srcc": 0.484, "oiq_plcc": 0.751, "oiq_srcc": 0.727},
}

# Table 4 — sampling scale
TABLE4_SAMPLING_SCALE = {
    (0.100, 0.050): {"jufe_plcc": 0.762, "jufe_srcc": 0.762, "oiq_plcc": 0.749, "oiq_srcc": 0.722},
    (0.200, 0.100): {"jufe_plcc": 0.782, "jufe_srcc": 0.781, "oiq_plcc": 0.766, "oiq_srcc": 0.749},
}

# Table 5 — component ablation
TABLE5_COMPONENTS = {
    "backbone_only": {"jufe_plcc": 0.687, "jufe_srcc": 0.673, "oiq_plcc": 0.695, "oiq_srcc": 0.662},
    "+PDFF": {"jufe_plcc": 0.760, "jufe_srcc": 0.757, "oiq_plcc": 0.751, "oiq_srcc": 0.732},
    "+HPA": {"jufe_plcc": 0.763, "jufe_srcc": 0.759, "oiq_plcc": 0.769, "oiq_srcc": 0.749},
    "full": {"jufe_plcc": 0.782, "jufe_srcc": 0.781, "oiq_plcc": 0.766, "oiq_srcc": 0.749},
}

# Table 8 — 2D-IQA adaptation
TABLE8_2D_IQA = {
    "HyperIQA": {"kadid_plcc": 0.872, "kadid_srcc": 0.869, "koniq_plcc": 0.900, "koniq_srcc": 0.915},
    "VU-BOIQA": {"kadid_plcc": 0.824, "kadid_srcc": 0.807, "koniq_plcc": 0.848, "koniq_srcc": 0.797},
}

DATABASES = ("CVIQ", "OIQA", "JUFE-10K", "OIQ-10K")


def ours_beats_assessor360_jufe() -> bool:
    o = TABLE1_OIQA_MAIN["VU-BOIQA"]
    a = TABLE1_OIQA_MAIN["Assessor360"]
    return o["jufe_srcc"] > a["jufe_srcc"]


def ours_lower_complexity_than_assessor360() -> bool:
    o = TABLE1_OIQA_MAIN["VU-BOIQA"]
    a = TABLE1_OIQA_MAIN["Assessor360"]
    return o["gflops"] < a["gflops"] and o["params_m"] < a["params_m"]


def pdff_improves_over_backbone() -> bool:
    b = TABLE5_COMPONENTS["backbone_only"]
    p = TABLE5_COMPONENTS["+PDFF"]
    return p["jufe_srcc"] > b["jufe_srcc"]


def cross_db_best_srcc() -> bool:
    v = TABLE2_CROSS_DB["VU-BOIQA"]
    return v["oiq_srcc"] >= max(m["oiq_srcc"] for k, m in TABLE2_CROSS_DB.items() if k != "VU-BOIQA")


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "table1": TABLE1_OIQA_MAIN,
        "table2": TABLE2_CROSS_DB,
        "table3": TABLE3_PATCH_COUNT,
        "table4": TABLE4_SAMPLING_SCALE,
        "table5": TABLE5_COMPONENTS,
        "table8": TABLE8_2D_IQA,
        "databases": list(DATABASES),
    }
