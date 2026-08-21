"""Reference metrics (Yin et al., arXiv:2509.24997, Table 1–2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoworld_x.config import (
    PANOEXPLORER_SCENES,
    PANOEXPLORER_VIDEOS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
)

# Table 1 — panoramic video generation (200 held-out eval clips)
TABLE1_PANORAMIC = {
    "360DVD": {"psnr": 10.66, "ssim": 0.35, "lpips": 0.73, "fid": 111.85, "fvd": 2049.21},
    "Imagine360": {"psnr": 11.62, "ssim": 0.39, "lpips": 0.591, "fid": 66.73, "fvd": 1830.46},
    "GenEX": {"psnr": 16.12, "ssim": 0.59, "lpips": 0.42, "fid": 42.22, "fvd": 1113.72},
    "PanoWorld-X": {"psnr": 19.34, "ssim": 0.63, "lpips": 0.24, "fid": 28.01, "fvd": 467.18},
}

# Table 1 — camera controllable (perspective crops)
TABLE1_CAMERA_CTRL = {
    "CameraCtrl": {
        "psnr": 11.56,
        "ssim": 0.38,
        "lpips": 0.61,
        "fid": 108.12,
        "fvd": 2017.95,
        "r_err": 0.097,
        "t_err": 0.245,
    },
    "AC3D": {
        "psnr": 13.77,
        "ssim": 0.49,
        "lpips": 0.52,
        "fid": 41.98,
        "fvd": 842.29,
        "r_err": 0.081,
        "t_err": 0.087,
    },
    "PanoWorld-X": {
        "psnr": 16.76,
        "ssim": 0.56,
        "lpips": 0.42,
        "fid": 38.63,
        "fvd": 586.51,
        "r_err": 0.061,
        "t_err": 0.073,
    },
}

# Table 2 — ablations
TABLE2_ABLATION = {
    "w/o Position Normalization": {
        "psnr": 17.11,
        "ssim": 0.55,
        "lpips": 0.32,
        "fid": 40.37,
        "fvd": 751.18,
        "r_err": 0.114,
        "t_err": 0.102,
    },
    "w/o Controllable Branch": {
        "psnr": 16.30,
        "ssim": 0.53,
        "lpips": 0.36,
        "fid": 38.71,
        "fvd": 769.42,
        "r_err": 0.102,
        "t_err": 0.152,
    },
    "w/o Sphere-Aware Attention": {
        "psnr": 17.59,
        "ssim": 0.56,
        "lpips": 0.27,
        "fid": 29.96,
        "fvd": 492.98,
        "r_err": 0.069,
        "t_err": 0.076,
    },
    "Full model": {
        "psnr": 19.34,
        "ssim": 0.63,
        "lpips": 0.24,
        "fid": 28.01,
        "fvd": 467.18,
        "r_err": 0.061,
        "t_err": 0.073,
    },
}

TRAINING_DETAIL = {
    "backbone": "CogVideoX-5B-I2V",
    "frames": 49,
    "native_train_res": "480×720",
    "erp_output_res": "480×960",
    "gpus": 8,
    "iter_controllable": 8000,
    "iter_sphere_attn": 2000,
    "first_frame": "FLUX + panorama LoRA (Yang et al.)",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "panoexplorer": {
            "videos": PANOEXPLORER_VIDEOS,
            "ue_scenes": PANOEXPLORER_SCENES,
        },
        "table1_panoramic": TABLE1_PANORAMIC,
        "table1_camera_controllable": TABLE1_CAMERA_CTRL,
        "table2_ablation": TABLE2_ABLATION,
        "training": TRAINING_DETAIL,
    }
