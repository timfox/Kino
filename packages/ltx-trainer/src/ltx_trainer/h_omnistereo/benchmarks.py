"""Reference tables from Jiang et al. (arXiv:2605.14963)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.h_omnistereo.config import (
    DATASET_NORMAL_IMAGES,
    DATASET_STEREO_PAIRS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
)

# Table I — omnidirectional stereo datasets
TABLE1_DATASETS: dict[str, dict[str, Any]] = {
    "Deep360": {"stereo_pairs": 18_000, "normal_gt": False, "camera_params": "finite set"},
    "360SD-Net": {"stereo_pairs": 4_000, "scenarios": ["indoor"]},
    "3D60": {"stereo_pairs": 25_000, "normal_gt": True},
    "Helvipad": {"stereo_pairs": 40_000, "source": "real-world", "precise_calibration": False},
    "Ours": {
        "stereo_pairs": DATASET_STEREO_PAIRS,
        "normal_gt": True,
        "precise_calibration": True,
        "camera_params": "arbitrary",
        "resolution": "1024x512",
        "source": "synthetic",
    },
}

# Table II — zero-shot stereo matching (MAE, RMSE, BP-1, BP-2, D1)
TABLE2_STEREO: dict[str, dict[str, float]] = {
    "360SD-Net": {
        "3d60_mae": 0.679,
        "3d60_warp_mae": 0.843,
        "mvs_gi_mae": 1.868,
        "3d60_bp1": 9.423,
    },
    "MODE": {"3d60_mae": 0.884, "3d60_warp_mae": 1.720, "mvs_gi_mae": 0.973, "3d60_bp1": 9.869},
    "360-IGEV-Stereo": {
        "3d60_mae": 7.000,
        "3d60_warp_mae": 9.665,
        "mvs_gi_mae": 10.985,
        "3d60_bp1": 86.969,
    },
    "DFI-OmniStereo": {
        "3d60_mae": 8.624,
        "3d60_warp_mae": 9.918,
        "mvs_gi_mae": 13.938,
        "3d60_bp1": 85.034,
    },
    "Ours": {
        "3d60_mae": 0.103,
        "3d60_rmse": 0.401,
        "3d60_bp1": 1.207,
        "3d60_bp2": 0.530,
        "3d60_d1": 0.326,
        "3d60_warp_mae": 0.163,
        "3d60_warp_bp1": 2.230,
        "mvs_gi_mae": 0.362,
        "mvs_gi_bp1": 5.635,
    },
}

TABLE3_PRIOR_ABLATION: dict[str, dict[str, float]] = {
    "DepthAnythingV2": {"3d60_mae": 0.115, "3d60_bp1": 1.445, "mvs_gi_mae": 0.447},
    "DA2": {"3d60_mae": 0.112, "3d60_bp1": 1.489, "mvs_gi_mae": 0.422},
    "Ours": {"3d60_mae": 0.103, "3d60_bp1": 1.207, "mvs_gi_mae": 0.362},
}

TABLE4_DATA_SCALING: list[dict[str, Any]] = [
    {"composition": "GRUtopia", "size_k": 522, "3d60_mae": 0.147, "3d60_bp1": 2.076},
    {"composition": "+ Chaotic", "size_k": 1539, "3d60_mae": 0.132},
    {"composition": "+ Realistic", "size_k": 2134, "3d60_mae": 0.124},
    {"composition": "+ HM3D", "size_k": 2833, "3d60_mae": 0.103, "3d60_bp1": 1.207},
]

TABLE5_DATASET_FOR_BASELINES: dict[str, dict[str, float]] = {
    "360SD-Net_original": {"3d60_mae": 0.679, "3d60_bp1": 9.423},
    "360SD-Net_ours_data": {"3d60_mae": 0.268, "3d60_bp1": 4.678},
    "MODE_original": {"3d60_mae": 0.884, "3d60_bp1": 9.869},
    "MODE_ours_data": {"3d60_mae": 0.276, "3d60_bp1": 4.767},
}

TABLE6_HEADING_ALIGNED: dict[str, dict[str, float]] = {
    "full_camera_coord": {"mae": 3.78, "rmse": 10.17, "delta5": 81.24},
    "full_heading_aligned": {"mae": 3.19, "rmse": 9.27, "delta5": 87.43},
    "crop_camera_coord": {"mae": 15.78, "rmse": 25.77, "delta5": 51.72},
    "crop_heading_aligned": {"mae": 3.66, "rmse": 9.36, "delta5": 84.62},
}

TABLE7_NORMAL_BASELINES: dict[str, dict[str, float]] = {
    "UniFuse": {"mae": 8.25, "rmse": 21.29, "delta5": 76.24},
    "PanoFormer": {"mae": 16.92, "rmse": 32.46, "delta5": 59.13},
    "OmniFusion": {"mae": 20.70, "rmse": 28.84, "delta5": 28.51},
    "HyperSphere": {"mae": 5.79, "rmse": 15.92, "delta5": 78.38},
    "PanoNormal": {"mae": 5.56, "rmse": 15.70, "delta5": 79.18},
    "Ours": {"mae": 3.83, "rmse": 12.12, "delta5": 86.49},
}

TABLE8_ODOMETRY: dict[str, dict[str, float]] = {
    "FoundationStereo": {"rte": 1.72e-2, "roe_deg": 2.42e-1, "rpe": 1.86e-2},
    "Ours_no_uncertainty": {"rte": 1.53e-2, "roe_deg": 1.63e-1, "rpe": 1.60e-2},
    "Ours": {"rte": 1.24e-2, "roe_deg": 1.27e-1, "rpe": 1.30e-2},
}

TRAINING = {
    "normal_images": DATASET_NORMAL_IMAGES,
    "stereo_pairs": DATASET_STEREO_PAIRS,
    "gpus": 8,
    "gpu_model": "NVIDIA L20",
    "steps": 600_000,
    "optimizer": "AdamW",
    "max_disparity": 256,
    "refine_iters": 22,
    "inference_s_512x1024": 0.47,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "project_url": PROJECT_URL,
        "table1_datasets": TABLE1_DATASETS,
        "table2_stereo": TABLE2_STEREO,
        "table3_prior_ablation": TABLE3_PRIOR_ABLATION,
        "table4_data_scaling": TABLE4_DATA_SCALING,
        "table5_baseline_on_our_data": TABLE5_DATASET_FOR_BASELINES,
        "table6_heading_aligned": TABLE6_HEADING_ALIGNED,
        "table7_normal": TABLE7_NORMAL_BASELINES,
        "table8_odometry": TABLE8_ODOMETRY,
        "training": TRAINING,
    }
