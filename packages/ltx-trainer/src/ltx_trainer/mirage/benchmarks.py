"""Paper table anchors for Mirage (arXiv:2606.09828)."""
from __future__ import annotations

from typing import Any

TABLE1_WORLDSORE: dict[str, dict[str, float]] = {
    "WonderJourney": {"average": 54.19, "static": 63.75, "dynamic": 44.63, "3d_cons": 80.60, "photo_cons": 79.03},
    "WonderWorld": {"average": 61.79, "static": 72.69, "dynamic": 50.88, "3d_cons": 86.87, "photo_cons": 85.56},
    "Voyager": {"average": 66.08, "static": 77.62, "dynamic": 54.53, "3d_cons": 81.56, "photo_cons": 85.99},
    "Spatia": {"average": 69.73, "static": 72.63, "dynamic": 66.82, "3d_cons": 86.40, "photo_cons": 89.10},
    "Wan2.1": {"average": 55.21, "static": 57.56, "dynamic": 52.85, "3d_cons": 78.74, "photo_cons": 78.36},
    "Mirage": {"average": 70.36, "static": 73.60, "dynamic": 67.11, "3d_cons": 92.21, "photo_cons": 93.95},
}

TABLE2_REALESTATE: dict[str, dict[str, float]] = {
    "SEVA": {"psnr": 13.07, "ssim": 0.515, "lpips": 0.445},
    "VMem": {"psnr": 14.62, "ssim": 0.522, "lpips": 0.426},
    "ViewCrafter": {"psnr": 15.78, "ssim": 0.580, "lpips": 0.396, "psnr_c": 14.79, "ssim_c": 0.481, "lpips_c": 0.365},
    "Voyager": {"psnr": 17.79, "ssim": 0.636, "lpips": 0.297, "psnr_c": 17.66, "ssim_c": 0.540, "lpips_c": 0.380},
    "Spatia": {"psnr": 18.58, "ssim": 0.646, "lpips": 0.254, "psnr_c": 19.38, "ssim_c": 0.579, "lpips_c": 0.213},
    "Mirage": {"psnr": 18.38, "ssim": 0.779, "lpips": 0.250, "psnr_c": 20.05, "ssim_c": 0.825, "lpips_c": 0.228},
}

TABLE3_ABLATION: dict[str, dict[str, float]] = {
    "Mirage (full)": {"avg": 70.36, "static": 73.60, "dynamic": 67.11, "3d_cons": 92.21, "photo_cons": 93.95},
    "Explicit RGB Point Cloud": {"avg": 67.71, "static": 70.49, "dynamic": 64.93, "3d_cons": 90.75, "photo_cons": 91.10},
    "Feature Upsample, Pixel Resolution Lift": {"avg": 60.85, "static": 62.41, "dynamic": 59.28, "3d_cons": 84.90, "photo_cons": 79.81},
    "No Dynamic Object Filter": {"avg": 61.20, "static": 62.69, "dynamic": 59.70, "3d_cons": 80.88, "photo_cons": 76.10},
    "Single Stage Training": {"avg": 63.18, "static": 65.15, "dynamic": 61.20, "3d_cons": 87.11, "photo_cons": 84.47},
}

TABLE4_DEPTH: dict[str, dict[str, float]] = {
    "DepthAnything 3": {"avg": 70.36, "static": 73.60, "dynamic": 67.11, "3d_cons": 92.21, "photo_cons": 93.95},
    "MapAnything": {"avg": 69.66, "static": 72.78, "dynamic": 66.53, "3d_cons": 91.89, "photo_cons": 93.32},
    "UniDepth": {"avg": 69.13, "static": 72.15, "dynamic": 66.10, "3d_cons": 91.63, "photo_cons": 92.79},
}

TABLE5_DOWNSAMPLE: dict[str, float] = {
    "bilinear": 42.53,
    "nearest": 47.78,
    "area": 53.72,
    "median": 52.22,
}


def table_worldscore() -> dict[str, dict[str, float]]:
    return dict(TABLE1_WORLDSORE)


def table_realestate10k() -> dict[str, dict[str, float]]:
    return dict(TABLE2_REALESTATE)


def table_ablation() -> dict[str, dict[str, float]]:
    return dict(TABLE3_ABLATION)


def table_depth_sources() -> dict[str, dict[str, float]]:
    return dict(TABLE4_DEPTH)


def table_depth_downsample() -> dict[str, float]:
    return dict(TABLE5_DOWNSAMPLE)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "worldscore": table_worldscore(),
        "realestate10k": table_realestate10k(),
        "ablation": table_ablation(),
        "depth_sources": table_depth_sources(),
        "depth_downsample": table_depth_downsample(),
        "paper": "2606.09828",
        "website": "https://aka.ms/latent-spatial-memory",
    }
