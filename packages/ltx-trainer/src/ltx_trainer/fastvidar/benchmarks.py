"""Reference metrics (Zhao et al., arXiv:2509.23733)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fastvidar.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL

# Table I — AHA ablation (HM3D)
TABLE1_AHA = {
    "No-Global (w+f)": {
        "abs_rel": 0.135,
        "rmse": 0.454,
        "log10": 0.181,
        "delta_125": 0.892,
        "time_ms": 34,
    },
    "AHA (w+f+g)": {
        "abs_rel": 0.111,
        "rmse": 0.384,
        "log10": 0.163,
        "delta_125": 0.904,
        "time_ms": 36,
    },
}

# Table II — fusion strategies
TABLE2_FUSION = {
    "No-fusion": {"abs_rel": 0.109, "rmse": 0.369, "log10": 0.149, "delta_125": 0.897},
    "Nearest": {"abs_rel": 0.113, "rmse": 0.384, "log10": 0.153, "delta_125": 0.898},
    "Weighted": {"abs_rel": 0.108, "rmse": 0.365, "log10": 0.146, "delta_125": 0.901},
    "Mean": {"abs_rel": 0.108, "rmse": 0.364, "log10": 0.146, "delta_125": 0.901},
}

# Table III — 2D-3D-S zero-shot
TABLE3_2D3DS = {
    "VGGT": {"abs_rel": 0.557, "rmse": 1.934, "log10": 0.396, "delta_125": 0.043, "time_ms": 120},
    "OmniStereo": {"abs_rel": 0.619, "rmse": 1.450, "log10": 0.154, "delta_125": 0.554, "time_ms": 66},
    "LightStereo": {"abs_rel": 0.125, "rmse": 0.667, "log10": 0.050, "delta_125": 0.851, "time_ms": 33},
    "FastViDAR": {"abs_rel": 0.119, "rmse": 0.433, "log10": 0.046, "delta_125": 0.929, "time_ms": 36},
}

DATASETS = {
    "HM3D_train_groups": 421_127,
    "HM3D_test_groups": 52_484,
    "2D3DS_zero_shot_groups": 6_000,
    "orin_nx_fps_fp16": 20,
    "vggt_speedup_640x320_4cam": 3.3,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "table1_aha": TABLE1_AHA,
        "table2_fusion": TABLE2_FUSION,
        "table3_2d3ds": TABLE3_2D3DS,
        "datasets": DATASETS,
    }
