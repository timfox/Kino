"""Paper-reported benchmark metrics (Tables 1–4, 3)."""

from __future__ import annotations

from typing import Any

# Table 1 — HDR-NeRF-Real + HDR-Plenoxels-Real (exp3, Ours† best overall)
TABLE1_EXP3_OURS_DAGGER: dict[str, dict[str, float]] = {
    "hdr_nerf_real_ldr_oe": {"psnr": 36.91, "ssim": 0.9777, "lpips": 0.009},
    "hdr_nerf_real_ldr_ne": {"psnr": 34.15, "ssim": 0.9737, "lpips": 0.012},
    "hdr_plenoxels_real_ldr_oe": {"psnr": 33.06, "ssim": 0.9592, "lpips": 0.025},
    "hdr_plenoxels_real_ldr_ne": {"psnr": 29.76, "ssim": 0.9389, "lpips": 0.034},
}

TABLE1_EXP3_HDR_GS: dict[str, dict[str, float]] = {
    "hdr_nerf_real_ldr_oe": {"psnr": 34.87, "ssim": 0.9697, "lpips": 0.021},
    "hdr_nerf_real_ldr_ne": {"psnr": 31.02, "ssim": 0.9636, "lpips": 0.029},
}

# Table 2 — HDR-NeRF-Syn (exp3)
TABLE2_EXP3: dict[str, dict[str, dict[str, float]]] = {
    "ours_dagger": {
        "ldr_oe": {"psnr": 44.26, "ssim": 0.9899, "lpips": 0.003},
        "ldr_ne": {"psnr": 43.19, "ssim": 0.9896, "lpips": 0.004},
        "hdr": {"psnr": 39.21, "ssim": 0.9768, "lpips": 0.010},
    },
    "hdr_gs": {
        "ldr_oe": {"psnr": 40.28, "ssim": 0.9781, "lpips": 0.018},
        "ldr_ne": {"psnr": 27.07, "ssim": 0.8744, "lpips": 0.127},
        "hdr": {"psnr": 17.51, "ssim": 0.6982, "lpips": 0.205},
    },
    "gausshdr_dagger": {
        "ldr_oe": {"psnr": 43.87, "ssim": 0.9899, "lpips": 0.004},
        "ldr_ne": {"psnr": 42.74, "ssim": 0.9894, "lpips": 0.004},
        "hdr": {"psnr": 39.08, "ssim": 0.9767, "lpips": 0.011},
    },
}

# Table 3 — efficiency (Ours 76 FPS vs HDR-GS 117 FPS)
TABLE3_EFFICIENCY: dict[str, dict[str, float]] = {
    "hdr_nerf": {"render_ms": 4189, "fps": 0.24, "train_min": 500, "memory_mb": 11049},
    "hdr_gs": {"render_ms": 9, "fps": 117, "train_min": 10, "memory_mb": 5014},
    "gausshdr": {"render_ms": 19, "fps": 53, "train_min": 28, "memory_mb": 5596},
    "ours": {"render_ms": 13, "fps": 76, "train_min": 15, "memory_mb": 3274},
    "ours_dagger": {"render_ms": 19, "fps": 53, "train_min": 18, "memory_mb": 3920},
}

# Table 4 — ablation (Ours† full model)
TABLE4_ABLATION: list[dict[str, Any]] = [
    {"config": "IE branch", "psnr_oe": 36.18, "psnr_ne": 33.38},
    {"config": "+ GI branch", "psnr_oe": 36.27, "psnr_ne": 33.46},
    {"config": "+ HDR-cons", "psnr_oe": 36.43, "psnr_ne": 33.84},
    {"config": "+ I-GS", "psnr_oe": 36.91, "psnr_ne": 34.15},
]

PSNR_GAIN_OVER_HDR_GS = 2.04  # abstract claim on HDR-NeRF-Syn


def psnr(pred, target) -> float:
    import torch

    if not isinstance(pred, torch.Tensor):
        raise TypeError("pred must be tensor")
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return 99.0
    import math

    return 10.0 * math.log10(1.0 / mse)
