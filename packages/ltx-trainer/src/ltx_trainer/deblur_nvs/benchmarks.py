"""Reference metrics from Shi et al. (arXiv:2606.01315)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deblur_nvs.config import (
    CONTEXT_DENOISE_STEPS,
    DEFAULT_CONTEXT_VIEWS,
    DL3DV_BLUR_PAIR_COUNT,
    DL3DV_SCENE_COUNT,
    INTERPOLATION_RATE,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
    RESOLUTION,
    TARGET_DENOISE_STEPS,
)

# Table I — dataset comparison
TABLE1_DATASETS: list[dict[str, Any]] = [
    {"name": "GoPro", "task": "Image Deblur", "scenes": 22, "pairs": 3214},
    {"name": "HIDE", "task": "Image Deblur", "scenes": 31, "pairs": 8422},
    {"name": "RealBlur", "task": "Image Deblur", "scenes": 232, "pairs": 4738},
    {"name": "REDS", "task": "Video Deblur", "scenes": 300, "pairs": "~30K"},
    {"name": "Deblur-NeRF", "task": "Deblur NVS", "scenes": 15, "pairs": None},
    {"name": "DL3DV-10K-Blur (Ours)", "task": "Deblur NVS", "scenes": DL3DV_SCENE_COUNT, "pairs": DL3DV_BLUR_PAIR_COUNT},
]

# Table II — DL3DV-Bench, 3 input views
TABLE2_DL3DV_BENCH: list[dict[str, Any]] = [
    {"method": "3DGS", "type": "scene-specific", "psnr": 14.769, "ssim": 0.527, "lpips": 0.422, "dists": 0.262, "fid": 233.40, "time": "32 min"},
    {"method": "BAGS", "type": "scene-specific", "psnr": 15.184, "ssim": 0.539, "lpips": 0.414, "dists": 0.259, "fid": 221.76, "time": "12 min"},
    {"method": "DA3", "type": "generalizable", "psnr": 10.945, "ssim": 0.357, "lpips": 0.581, "dists": 0.337, "fid": 275.52, "time": "0.03 s"},
    {"method": "GLD", "type": "generalizable", "psnr": 11.461, "ssim": 0.385, "lpips": 0.503, "dists": 0.235, "fid": 150.90, "time": "9.76 s"},
    {"method": "Ours", "type": "generalizable", "psnr": 15.549, "ssim": 0.441, "lpips": 0.367, "dists": 0.174, "fid": 101.36, "time": "0.60 s"},
]

# Table III — DeblurNeRF-Real (3/6/9 views subset)
TABLE3_DEBLURNERF_REAL: dict[int, list[dict[str, Any]]] = {
    3: [
        {"method": "3DGS", "type": "scene-specific", "psnr": 18.454, "ssim": 0.512, "lpips": 0.400, "dists": 0.253, "fid": 166.348, "time": "12 min"},
        {"method": "BAGS", "type": "scene-specific", "psnr": 18.813, "ssim": 0.527, "lpips": 0.385, "dists": 0.241, "fid": 151.080, "time": "30 min"},
        {"method": "DA3", "type": "generalizable", "psnr": 13.610, "ssim": 0.356, "lpips": 0.608, "dists": 0.411, "fid": 306.244, "time": "0.03 s"},
        {"method": "GLD", "type": "generalizable", "psnr": 12.868, "ssim": 0.329, "lpips": 0.570, "dists": 0.281, "fid": 166.513, "time": "9.77 s"},
        {"method": "Ours", "type": "generalizable", "psnr": 17.132, "ssim": 0.433, "lpips": 0.335, "dists": 0.142, "fid": 79.648, "time": "0.60 s"},
    ],
    6: [
        {"method": "Ours", "type": "generalizable", "psnr": 17.893, "ssim": 0.464, "lpips": 0.301, "dists": 0.129, "fid": 73.571, "time": "0.80 s"},
    ],
    9: [
        {"method": "Ours", "type": "generalizable", "psnr": 18.091, "ssim": 0.475, "lpips": 0.290, "dists": 0.125, "fid": 70.221, "time": "1.04 s"},
    ],
}

# Table IV — DeblurNeRF-Blender, 3 views
TABLE4_DEBLURNERF_BLENDER: list[dict[str, Any]] = [
    {"method": "3DGS", "type": "scene-specific", "psnr": 17.873, "ssim": 0.482, "lpips": 0.406, "dists": 0.249, "fid": 177.976, "time": "12 min"},
    {"method": "BAGS", "type": "scene-specific", "psnr": 18.021, "ssim": 0.485, "lpips": 0.404, "dists": 0.247, "fid": 173.374, "time": "30 min"},
    {"method": "DA3", "type": "generalizable", "psnr": 11.745, "ssim": 0.282, "lpips": 0.665, "dists": 0.456, "fid": 344.404, "time": "0.04 s"},
    {"method": "GLD", "type": "generalizable", "psnr": 16.009, "ssim": 0.403, "lpips": 0.451, "dists": 0.243, "fid": 162.288, "time": "10.07 s"},
    {"method": "Ours", "type": "generalizable", "psnr": 16.049, "ssim": 0.387, "lpips": 0.340, "dists": 0.149, "fid": 103.528, "time": "0.66 s"},
]

# Table V — ablation (DeblurNeRF-Real, 3 views)
TABLE5_ABLATION: list[dict[str, Any]] = [
    {"method": "w/o CL", "psnr": 16.989, "ssim": 0.437, "lpips": 0.345, "dists": 0.149, "fid": 82.828},
    {"method": "w/o LoRA", "psnr": 17.122, "ssim": 0.433, "lpips": 0.336, "dists": 0.145, "fid": 83.148},
    {"method": "Ours", "psnr": 17.132, "ssim": 0.433, "lpips": 0.335, "dists": 0.142, "fid": 79.648},
]

# Table VI — Uformer+GLD cascaded baseline
TABLE6_CASCADED: list[dict[str, Any]] = [
    {"method": "GLD", "psnr": 12.868, "ssim": 0.329, "lpips": 0.570, "dists": 0.281, "fid": 166.513},
    {"method": "Uformer+GLD", "psnr": 12.643, "ssim": 0.335, "lpips": 0.602, "dists": 0.252, "fid": 151.051},
    {"method": "Ours", "psnr": 17.132, "ssim": 0.433, "lpips": 0.335, "dists": 0.142, "fid": 79.648},
]

TRAINING = {
    "backbone": "GLD single-level DA3 latent (no cascade)",
    "dataset": "DL3DV-10K-Blur",
    "resolution": f"{RESOLUTION[0]}×{RESOLUTION[1]}",
    "context_denoise_steps": CONTEXT_DENOISE_STEPS,
    "target_denoise_steps": TARGET_DENOISE_STEPS,
    "gpu": "single 48 GB RTX 4090",
    "train_time": "2–3 days",
    "context_views_default": DEFAULT_CONTEXT_VIEWS,
    "blur_synthesis": f"{INTERPOLATION_RATE}× interp, N∈{5,7,9,11}",
}


def table2_ours() -> dict[str, Any]:
    return next(r for r in TABLE2_DL3DV_BENCH if r["method"] == "Ours")


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL, "project": PROJECT_URL},
        "table1_datasets": TABLE1_DATASETS,
        "table2_dl3dv_bench": TABLE2_DL3DV_BENCH,
        "table3_deblurnerf_real": TABLE3_DEBLURNERF_REAL,
        "table4_deblurnerf_blender": TABLE4_DEBLURNERF_BLENDER,
        "table5_ablation": TABLE5_ABLATION,
        "table6_cascaded": TABLE6_CASCADED,
        "training": TRAINING,
    }
