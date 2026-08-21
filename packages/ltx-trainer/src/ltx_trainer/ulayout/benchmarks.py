"""Paper tables (Lee et al., arXiv:2503.21562)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ulayout.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — PanoContext + whole Stanford 2D-3D + LSUN
TABLE1_PANO_ST2D_LSUN = {
    "LGT-Net": {"pano_2d_iou": 87.88, "pano_3d_iou": 85.16, "lsun_ceiling": 23.49, "lsun_floor": 42.69},
    "DOP-Net": {"pano_2d_iou": 88.02, "pano_3d_iou": 85.46, "lsun_ceiling": 33.02, "lsun_floor": 41.10},
    "LSUN-ROOM": {"lsun_ceiling": 76.59, "lsun_floor": 73.62},
    "FUSING": {"lsun_ceiling": 80.68, "lsun_floor": 80.03},
    "Ours": {"pano_2d_iou": 88.70, "pano_3d_iou": 86.04, "lsun_ceiling": 83.12, "lsun_floor": 80.12},
}

# Table 2 — Stanford 2D-3D + whole PanoContext + LSUN
TABLE2_ST2D_PANO_LSUN = {
    "LGT-Net": {"pano_2d_iou": 88.09, "pano_3d_iou": 86.03, "lsun_ceiling": 16.91, "lsun_floor": 30.94},
    "DOP-Net": {"pano_2d_iou": 87.73, "pano_3d_iou": 85.58, "lsun_ceiling": 19.88, "lsun_floor": 30.81},
    "Ours": {"pano_2d_iou": 88.64, "pano_3d_iou": 86.90, "lsun_ceiling": 83.30, "lsun_floor": 80.11},
}

# Table 3 — MatterportLayout + LSUN
TABLE3_MATTERPORT_LSUN = {
    "LGT-Net": {"pano_2d_iou": 83.52, "pano_3d_iou": 81.11, "lsun_ceiling": 6.86, "lsun_floor": 26.06},
    "DOP-Net": {"pano_2d_iou": 84.11, "pano_3d_iou": 81.70, "lsun_ceiling": 8.78, "lsun_floor": 31.17},
    "LSUN-ROOM": {"lsun_ceiling": 76.59, "lsun_floor": 73.62},
    "FUSING": {"lsun_ceiling": 80.68, "lsun_floor": 80.03},
    "Ours": {"pano_2d_iou": 84.05, "pano_3d_iou": 81.84, "lsun_ceiling": 83.61, "lsun_floor": 80.25},
}

# Table 4 — ablation (MatterportLayout + LSUN)
TABLE4_ABLATION = {
    "only_panorama": {"pano_2d_iou": 83.08, "pano_3d_iou": 80.83, "lsun_ceiling": 3.96, "lsun_floor": 0.57},
    "only_perspective": {"pano_2d_iou": 31.71, "pano_3d_iou": 29.46, "lsun_ceiling": 74.84, "lsun_floor": 75.09},
    "w/o_vertical_shift": {"pano_2d_iou": 83.35, "pano_3d_iou": 81.32, "lsun_ceiling": 78.22, "lsun_floor": 72.20},
    "Ours": {"pano_2d_iou": 84.05, "pano_3d_iou": 81.84, "lsun_ceiling": 83.61, "lsun_floor": 80.25},
}

# Table 5 — efficient feature extraction (perspective)
TABLE5_EFFICIENCY = {
    "resnet50": {
        "original": {"time_ms": 1.88, "memory_mb": 75.00, "gflops": 107.93},
        "efficient": {"time_ms": 1.88, "memory_mb": 19.22, "gflops": 26.98},
    },
    "conv1d": {
        "original": {"time_ms": 3.48, "memory_mb": 1.25, "gflops": 63.86},
        "efficient": {"time_ms": 1.17, "memory_mb": 0.31, "gflops": 15.97},
    },
}

DATASETS = ("PanoContext", "Stanford2D3D", "MatterportLayout", "LSUN")
PANORAMA_BASELINES = ("LGT-Net", "DOP-Net")
PERSPECTIVE_BASELINES = ("LSUN-ROOM", "FUSING")


def ours_beats_pano_baseline(table: dict[str, dict[str, float]], baseline: str) -> bool:
    o = table["Ours"]
    b = table[baseline]
    return o["pano_2d_iou"] >= b["pano_2d_iou"] and o["pano_3d_iou"] >= b["pano_3d_iou"]


def ours_beats_lsun_room(table: dict[str, dict[str, float]]) -> bool:
    o, r = table["Ours"], table.get("LSUN-ROOM", table.get("FUSING", {}))
    if "lsun_ceiling" not in r:
        return False
    return o["lsun_ceiling"] > r["lsun_ceiling"] and o["lsun_floor"] > r["lsun_floor"]


def joint_training_wins_ablation() -> bool:
    a = TABLE4_ABLATION
    return (
        a["Ours"]["lsun_ceiling"] > a["only_panorama"]["lsun_ceiling"]
        and a["Ours"]["pano_2d_iou"] > a["only_perspective"]["pano_2d_iou"]
        and a["Ours"]["lsun_ceiling"] > a["w/o_vertical_shift"]["lsun_ceiling"]
    )


def efficient_reduces_flops() -> bool:
    r = TABLE5_EFFICIENCY["resnet50"]
    c = TABLE5_EFFICIENCY["conv1d"]
    return r["efficient"]["gflops"] < 0.3 * r["original"]["gflops"] and c["efficient"]["gflops"] < 0.3 * c["original"]["gflops"]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "table1": TABLE1_PANO_ST2D_LSUN,
        "table2": TABLE2_ST2D_PANO_LSUN,
        "table3": TABLE3_MATTERPORT_LSUN,
        "table4": TABLE4_ABLATION,
        "table5": TABLE5_EFFICIENCY,
        "datasets": list(DATASETS),
    }
