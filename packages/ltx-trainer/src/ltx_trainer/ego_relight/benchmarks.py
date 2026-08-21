"""Reference numbers from paper tables."""

from __future__ import annotations

from typing import Any

# Table 4 — Ours row per subject (PSNR, SSIM×100, LPIPS×100, FID)
TABLE4_OURS: dict[int, dict[str, float]] = {
    1: {"psnr": 34.81, "ssim": 92.46, "lpips": 8.56, "fid": 31.90},
    2: {"psnr": 36.02, "ssim": 89.33, "lpips": 5.90, "fid": 23.20},
    3: {"psnr": 35.47, "ssim": 88.63, "lpips": 5.23, "fid": 31.81},
    4: {"psnr": 33.79, "ssim": 87.07, "lpips": 6.89, "fid": 42.93},
}

# Table 2 — egocentric reconstruction (cm)
TABLE2_P2S_CM: dict[str, float] = {
    "ours_full_body": 1.23,
    "ours_visible": 1.12,
    "egoavatar": 1.11,
    "ddc": 1.24,
}

# Table 5 ablation (Subject aggregate, Ours r=32)
TABLE5_ABLATION: dict[str, dict[str, float]] = {
    "ours_r32": {"psnr": 34.81, "ssim": 92.46, "lpips": 8.56, "fid": 31.90},
    "w/o_geolift": {"psnr": 34.35, "fid": 41.25},
    "w/o_diffuse": {"psnr": 19.00, "fid": 207.42},
    "w/o_specular": {"psnr": 34.61, "fid": 76.25},
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table4_ours": TABLE4_OURS,
        "table2_p2s_cm": TABLE2_P2S_CM,
        "table5_ablation": TABLE5_ABLATION,
        "lightstage_lights": 331,
        "lightstage_cameras_train": 37,
    }
