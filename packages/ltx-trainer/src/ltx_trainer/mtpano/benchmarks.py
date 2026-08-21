"""Reference metrics (Zhang et al. arXiv:2602.05330)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mtpano.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — Structured3D (selected rows + Ours)
TABLE1_STRUCTURED3D: list[dict[str, Any]] = [
    {"method": "InvPT", "mIoU": 70.02, "AbsRel": 0.0421, "RMSE": 0.1515, "norm_mean": 5.991},
    {"method": "BridgeNet", "mIoU": 70.13, "AbsRel": 0.0418, "RMSE": 0.1492, "norm_mean": 5.988},
    {"method": "TaskPrompter", "mIoU": 70.95, "AbsRel": 0.0414, "RMSE": 0.1428, "norm_mean": 5.962},
    {"method": "Ours", "mIoU": 75.66, "AbsRel": 0.0248, "RMSE": 0.0968, "norm_mean": 3.850},
]

# Table 2 — Stanford2D3D
TABLE2_STANFORD: list[dict[str, Any]] = [
    {"method": "TaskPrompter", "mIoU": 57.55, "AbsRel": 0.1171, "RMSE": 0.5792, "norm_mean": 12.390},
    {"method": "BridgeNet", "mIoU": 57.12, "AbsRel": 0.1198, "RMSE": 0.5834, "norm_mean": 12.840},
    {"method": "Ours", "mIoU": 69.47, "AbsRel": 0.0675, "RMSE": 0.4317, "norm_mean": 9.706},
]

# Table 3 — Matterport3D ablation (ViT-S)
TABLE3_ABLATION: list[dict[str, Any]] = [
    {"variant": "MTL baseline", "mIoU": 21.94, "RMSE": 0.5427, "norm_mean": 13.774, "delta_mtl_pct": -3.64},
    {"variant": "PD-BridgeNet (Full)", "mIoU": 26.89, "RMSE": 0.5234, "norm_mean": 12.637, "delta_mtl_pct": 7.21},
]

# Table 4 — Matterport3D (Ours row; full table in paper)
TABLE4_MATTERPORT: list[dict[str, Any]] = [
    {"method": "InvPT", "mIoU": 27.31, "AbsRel": 0.1145, "norm_mean": 18.072},
    {"method": "MultiPanoWise", "mIoU": 25.52, "AbsRel": 0.1360, "norm_mean": 20.739},
    {"method": "Ours", "mIoU": 39.11, "AbsRel": 0.1080, "RMSE": 0.4339, "norm_mean": 9.970},
]

# Table 5 — SynPASS / Deep360 / PanoSUNCG (Ours row)
TABLE5_EXTENDED: dict[str, Any] = {
    "SynPASS_val_mIoU": 49.56,
    "SynPASS_test_mIoU": 44.71,
    "Deep360_AbsRel": 0.0224,
    "Deep360_RMSE": 3.1707,
    "PanoSUNCG_AbsRel": 0.0312,
    "PanoSUNCG_RMSE": 0.1331,
}

# Table 6 — task combinations on Matterport3D (ViT-S)
TABLE6_TASK_COMBOS: list[dict[str, Any]] = [
    {"method": "STL", "mIoU": 24.14, "RMSE": 0.5523, "norm_mean": 13.302},
    {"method": "Semseg+Depth+Normals", "mIoU": 26.89, "RMSE": 0.5234, "norm_mean": 12.637},
]

# Table 7 — backbone init (Matterport3D, ViT-S)
TABLE7_BACKBONE_INIT: list[dict[str, Any]] = [
    {"init": "Random", "mIoU": 12.76, "RMSE": 0.6739, "norm_mean": 15.113},
    {"init": "ImageNet+Warmup", "mIoU": 26.89, "RMSE": 0.5234, "norm_mean": 12.637},
    {"init": "DINOv3+Warmup", "mIoU": 28.58, "RMSE": 0.4894, "norm_mean": 12.387},
]

# Table 8 — Structured3D ablation (DINOv3-S)
TABLE8_STRUCTURED3D_ABLATION: list[dict[str, Any]] = [
    {"variant": "MTL baseline", "mIoU": 65.17, "RMSE": 0.1501, "norm_mean": 5.522, "delta_mtl_pct": 10.88},
    {"variant": "Full MTPano", "mIoU": 66.64, "RMSE": 0.1432, "norm_mean": 5.474, "delta_mtl_pct": 13.07},
]

# Table 9 — auxiliary tasks (Matterport3D)
TABLE9_AUXILIARY: list[dict[str, Any]] = [
    {"aux": "None", "mIoU": 24.38, "RMSE": 0.5278, "norm_mean": 13.164},
    {"aux": "+Point+EDF+Grad", "mIoU": 26.89, "RMSE": 0.5234, "norm_mean": 12.637},
]

# Fig. 4 — data scalability on Stanford2D3D (mIoU / RMSE / normal mean)
FIG4_DATA_SCALE: list[dict[str, Any]] = [
    {"train_panos_k": 0, "mIoU": 68.5, "depth_RMSE": 0.384, "norm_mean": 11.5},
    {"train_panos_k": 17, "mIoU": 69.8, "depth_RMSE": 0.349, "norm_mean": 10.1},
    {"train_panos_k": 140, "mIoU": 70.4, "depth_RMSE": 0.329, "norm_mean": 9.58},
]


def table1_ours() -> dict[str, Any]:
    return next(r for r in TABLE1_STRUCTURED3D if r["method"] == "Ours")


def table4_ours() -> dict[str, Any]:
    return next(r for r in TABLE4_MATTERPORT if r["method"] == "Ours")


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "table1_structured3d": TABLE1_STRUCTURED3D,
        "table2_stanford": TABLE2_STANFORD,
        "table3_ablation": TABLE3_ABLATION,
        "table4_matterport": TABLE4_MATTERPORT,
        "table5_extended": TABLE5_EXTENDED,
        "table6_task_combos": TABLE6_TASK_COMBOS,
        "table7_backbone_init": TABLE7_BACKBONE_INIT,
        "table8_structured3d_ablation": TABLE8_STRUCTURED3D_ABLATION,
        "table9_auxiliary": TABLE9_AUXILIARY,
        "fig4_data_scale": FIG4_DATA_SCALE,
    }
