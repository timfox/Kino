"""Paper anchors — S³U-SAR (Yin et al., arXiv:2606.06847)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06847"
PAPER_TITLE = (
    "Physics-Driven Semantic Scattering Structure Understanding of Aircraft Target in SAR Images"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
GITHUB_URL = "https://github.com/YYF121/S3U-SAR"

NUM_KEYPOINTS = 10
INPUT_SIZE = (256, 192)  # H × W

AIRCRAFT_CATEGORIES: tuple[str, ...] = (
    "A220",
    "A320/321",
    "A330",
    "ARJ21",
    "Boeing 737",
    "Boeing 787",
    "others",
)

DATASET_KP_SAR = {
    "name": "KP-SAR-Aircraft-1.0",
    "samples": 2990,
    "resolution_m": 1.0,
    "sensor": "Gaofen-3",
    "categories": len(AIRCRAFT_CATEGORIES),
}

# Visibility labels — Eq. 3
VISIBILITY_SALIENT = 2
VISIBILITY_DEGRADED = 1
VISIBILITY_INVALID = 0

# Training hyperparameters (Sec. IV-B)
TRAIN_DEFAULTS = {
    "backbone": "HRNet-w32",
    "epochs": 20,
    "lr": 1e-3,
    "optimizer": "Adam",
    "temperature_start": 1.0,
    "temperature_end": 0.3,
    "alpha": 0.8,
    "mu": 0.4,
    "beta": 0.15,
}

# Table I — S3U-SAR HRNet-W32 vs baselines (AP %)
TABLE1_S3U_HRNET_W32 = {
    "AP": 59.3,
    "AP50": 85.4,
    "AP75": 66.8,
    "APM": 58.4,
    "APL": 73.0,
    "AR": 71.9,
    "AR50": 91.6,
    "AR75": 79.8,
}

TABLE1_BASELINES_AP: tuple[dict[str, str | float], ...] = (
    {"method": "HRNet-W32", "AP": 54.7},
    {"method": "HRNet-W48", "AP": 55.3},
    {"method": "ViTPose-Base", "AP": 55.5},
    {"method": "SimCC", "AP": 36.7},
    {"method": "DiffusionPose", "AP": 36.2},
    {"method": "ProbPose", "AP": 46.7},
    {"method": "S3U-SAR HRNet-W32", "AP": 59.3},
)

# Table II ablation (AP %)
TABLE2_ABLATION: tuple[dict[str, str | float | bool], ...] = (
    {"Lmse": True, "Lhetero": False, "Ltopo": False, "Lentropy": False, "AP": 55.7, "AP75": 61.4},
    {"Lmse": True, "Lhetero": True, "Ltopo": False, "Lentropy": False, "AP": 56.7, "AP75": 62.9},
    {"Lmse": True, "Lhetero": False, "Ltopo": True, "Lentropy": False, "AP": 58.2, "AP75": 65.0},
    {"Lmse": True, "Lhetero": True, "Ltopo": True, "Lentropy": False, "AP": 57.7, "AP75": 65.4},
    {"Lmse": True, "Lhetero": True, "Ltopo": True, "Lentropy": True, "AP": 59.3, "AP75": 66.8},
)

# Table III cross-category AP (unseen)
TABLE3_CROSS_CATEGORY: tuple[dict[str, str | float], ...] = (
    {"category": "A330", "AP": 51.6, "AP50": 85.3},
    {"category": "A320", "AP": 34.8, "AP50": 71.5},
    {"category": "Boeing 737", "AP": 38.2, "AP50": 76.5},
    {"category": "Boeing 787", "AP": 46.6, "AP50": 81.0},
)

# Table IV orientation estimation
TABLE4_ORIENTATION: tuple[dict[str, str | float], ...] = (
    {"method": "ResNet-18", "P1": 8.20, "P5": 38.62, "MAE": 17.31},
    {"method": "ResNet-50", "P1": 12.95, "P5": 46.88, "MAE": 16.31},
    {"method": "ViT", "P1": 11.72, "P5": 40.62, "MAE": 17.37},
    {"method": "S3U-SAR (Ours)", "P1": 30.13, "P5": 68.53, "MAE": 13.15},
)

ORIENTATION_GAIN_P1 = 17.07  # pp over second-best (ResNet-18 P1 baseline narrative uses ResNet-50 as strong)
ORIENTATION_GAIN_P5 = 19.31

# Semantic keypoint indices (Fig. 2 — component names stub)
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose_tip",
    "nose_root",
    "wing_tip_left",
    "wing_tip_right",
    "wing_root_left",
    "wing_root_right",
    "tail_tip",
    "tail_root",
    "engine_left",
    "engine_right",
)
