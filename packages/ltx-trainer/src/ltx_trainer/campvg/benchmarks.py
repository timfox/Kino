"""Paper Tables 1–3 (Ji et al., arXiv:2509.19979)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.campvg.config import DOI, PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — baseline comparison
TABLE1_BASELINES = {
    "MotionCtrl": {
        "lpips": 0.1741,
        "ssim": 0.6008,
        "psnr": 29.51,
        "faed": 0.2993,
        "fvd": 94.84,
        "vbench": 73.75,
        "vbench_aesthetic": 0.4878,
        "vbench_subject": 0.8706,
        "vbench_flicker": 0.9256,
    },
    "CameraCtrl": {
        "lpips": 0.1816,
        "ssim": 0.5926,
        "psnr": 29.29,
        "faed": 0.3967,
        "fvd": 151.08,
        "vbench": 135.90,
        "vbench_aesthetic": 0.4832,
        "vbench_subject": 0.8593,
        "vbench_flicker": 0.9241,
    },
    "CamI2V": {
        "lpips": 0.1867,
        "ssim": 0.5887,
        "psnr": 29.24,
        "faed": 0.2621,
        "fvd": 91.83,
        "vbench": 76.25,
        "vbench_aesthetic": 0.4931,
        "vbench_subject": 0.8755,
        "vbench_flicker": 0.9302,
    },
    "CamPVG": {
        "lpips": 0.1480,
        "ssim": 0.6544,
        "psnr": 30.05,
        "faed": 0.1066,
        "fvd": 66.24,
        "vbench": 56.34,
        "vbench_aesthetic": 0.5043,
        "vbench_subject": 0.9000,
        "vbench_flicker": 0.9339,
    },
}

# Table 2 — component ablation
TABLE2_ABLATION = {
    "CamPVG_full": {
        "pano_plucker": True,
        "sph_epipolar": True,
        "rand_cond": True,
        "lpips": 0.1480,
        "ssim": 0.6544,
        "psnr": 30.05,
        "faed": 0.1066,
        "fvd": 66.24,
        "vbench": 56.34,
    },
    "w/o_sph_epipolar": {
        "pano_plucker": True,
        "sph_epipolar": False,
        "rand_cond": True,
        "lpips": 0.1865,
        "ssim": 0.5998,
        "psnr": 29.48,
        "faed": 0.2704,
        "fvd": 87.14,
        "vbench": 77.13,
    },
    "w/o_pano_plucker": {
        "pano_plucker": False,
        "sph_epipolar": True,
        "rand_cond": True,
        "lpips": 0.3278,
        "ssim": 0.4868,
        "psnr": 28.85,
        "faed": 0.4954,
        "fvd": 124.93,
        "vbench": 101.03,
    },
    "w/o_rand_cond": {
        "pano_plucker": True,
        "sph_epipolar": True,
        "rand_cond": False,
        "lpips": 0.4144,
        "ssim": 0.4794,
        "psnr": 28.51,
        "faed": 0.7591,
        "fvd": 259.03,
        "vbench": 157.92,
    },
}

# Table 3 — epipolar sample count K
TABLE3_EPIPOLAR_K = {
    100: {"lpips": 0.1525, "ssim": 0.6398, "psnr": 29.91, "faed": 0.1240, "fvd": 72.68, "vbench": 63.78},
    150: {"lpips": 0.1499, "ssim": 0.6429, "psnr": 29.98, "faed": 0.1146, "fvd": 71.22, "vbench": 63.56},
    200: {"lpips": 0.1500, "ssim": 0.6457, "psnr": 29.94, "faed": 0.1141, "fvd": 67.66, "vbench": 57.63},
    250: {"lpips": 0.1480, "ssim": 0.6544, "psnr": 30.05, "faed": 0.1066, "fvd": 66.24, "vbench": 56.34},
    300: {"lpips": 0.1486, "ssim": 0.6458, "psnr": 29.99, "faed": 0.1130, "fvd": 75.43, "vbench": 68.37},
}

DATASETS = {
    "source": "3D-FRONT cubemap → ERP",
    "scenes": 5616,
    "clip_frames": 16,
    "resolution": "256×512",
    "train_epochs": 300,
    "gpus": 8,
    "batch_size": 16,
    "base_model": "DynamiCrafter",
}

USER_STUDY = {
    "MotionCtrl": {"camera": 1.895, "condition": 1.898, "quality": 1.788},
    "CameraCtrl": {"camera": 1.835, "condition": 1.743, "quality": 1.895},
    "CamI2V": {"camera": 2.753, "condition": 2.763, "quality": 2.770},
    "CamPVG": {"camera": 3.518, "condition": 3.598, "quality": 3.548},
    "CamPVG_realworld_cond": 3.817,
    "CamPVG_realworld_quality": 3.783,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "doi": DOI,
        "table1_baselines": TABLE1_BASELINES,
        "table2_ablation": TABLE2_ABLATION,
        "table3_epipolar_k": TABLE3_EPIPOLAR_K,
        "datasets": DATASETS,
        "user_study": USER_STUDY,
    }
