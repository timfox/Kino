"""Paper benchmark metrics (Tables 1–4)."""

from __future__ import annotations

TABLE1_LH3D_KU: dict[str, dict[str, float]] = {
    "lightharmony3d": {"psnr": 24.03, "ssim": 0.832, "lpips": 0.200},
    "gigs": {"psnr": 17.33, "ssim": 0.698, "lpips": 0.334},
    "gaussian_editor": {"psnr": 21.56, "ssim": 0.812, "lpips": 0.215},
    "mv_colight": {"psnr": 15.99, "ssim": 0.747, "lpips": 0.256},
    "gaslight": {"psnr": 20.41, "ssim": 0.812, "lpips": 0.224},
}

TABLE3_LH3D_BLENDER: dict[str, dict[str, float]] = {
    "lightharmony3d": {"psnr": 23.987, "ssim": 0.744, "lpips": 0.335},
    "gigs": {"psnr": 20.110, "ssim": 0.690, "lpips": 0.406},
    "gaslight": {"psnr": 19.171, "ssim": 0.626, "lpips": 0.399},
}

TABLE2_VQA_MIPNERF360: dict[str, dict[str, float]] = {
    "lightharmony3d": {"positive": 0.528, "negative": 0.208, "ratio": 0.751},
    "gaslight": {"positive": 0.472, "negative": 0.541, "ratio": 0.457},
    "3dgs": {"positive": 0.351, "negative": 0.400, "ratio": 0.501},
}

TABLE4_ABLATION: list[dict[str, float | str]] = [
    {"variant": "w/o Exposure Fusion", "psnr": 23.198, "ssim": 0.8173, "lpips": 0.2190},
    {"variant": "w/o Shadow Ratio", "psnr": 22.490, "ssim": 0.8217, "lpips": 0.2093},
    {"variant": "w/o Ray-Decoupled Shader", "psnr": 23.543, "ssim": 0.8317, "lpips": 0.2007},
    {"variant": "Full Model", "psnr": 24.032, "ssim": 0.8318, "lpips": 0.2004},
]
