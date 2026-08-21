"""Reference metrics from LumiVid paper (arXiv:2604.11788)."""

from __future__ import annotations

from typing import Any

PAPER_ARXIV = "2604.11788"
PAPER_TITLE = "HDR Video Generation via Latent Alignment with Logarithmic Encoding"

# Table 1 — frozen VAE roundtrip by encoding
TABLE1_VAE_ROUNDTRIP: dict[str, dict[str, float]] = {
    "logc3": {"kl_px": 0.215, "kl_lat": 0.007, "ssim": 0.9747, "pu21_psnr": 34.70, "pu21_ssim": 0.9677},
    "pq": {"kl_px": 0.289, "kl_lat": 0.011, "ssim": 0.9749, "pu21_psnr": 35.73, "pu21_ssim": 0.9668},
    "hlg": {"kl_px": 0.373, "kl_lat": 3.419, "ssim": 0.8658, "pu21_psnr": 22.86, "pu21_ssim": 0.8653},
    "aces": {"kl_px": 3.338, "kl_lat": 0.016, "ssim": 0.7823, "pu21_psnr": 16.42, "pu21_ssim": 0.6393},
}

# Table 2 — vs baselines (ARRI + UPIQ excerpts)
TABLE2_BASELINES_ARRI: dict[str, dict[str, float]] = {
    "lumivid": {"pu21_psnr": 36.20, "lpips": 0.020, "jod": 7.86},
    "hdrtvnet": {"pu21_psnr": 26.48, "lpips": 0.089, "jod": 6.94},
    "x2hdr": {"pu21_psnr": 20.68, "lpips": 0.250, "jod": 3.54},
}

TABLE2_BASELINES_UPIQ: dict[str, dict[str, float]] = {
    "lumivid": {"pu21_psnr": 30.05, "lpips": 0.071, "jod": 8.22},
    "hdrtvnet": {"pu21_psnr": 22.59, "lpips": 0.071, "jod": 4.48},
    "lediff": {"pu21_psnr": 14.94, "lpips": 0.212, "jod": 0.40},
    "x2hdr": {"pu21_psnr": 17.47, "lpips": 0.177, "jod": 6.06},
}

# Table 3 — temporal stability ARRI
TABLE3_TEMPORAL: dict[str, dict[str, float]] = {
    "lumivid": {"f2f_psnr": 45.63, "flicker": 0.0245, "jod": 7.86},
    "hdrtvnet": {"f2f_psnr": 44.59, "flicker": 0.0162, "jod": 6.94},
    "x2hdr": {"f2f_psnr": 36.36, "flicker": 0.1630, "jod": 3.54},
}

# Table 4 — encoding ablation
TABLE4_ENCODING: dict[str, dict[str, float]] = {
    "logc3": {"kl_sdr": 0.302, "pu21_psnr": 36.97, "lpips": 0.020, "jod": 7.86},
    "pq": {"kl_sdr": 0.377, "pu21_psnr": 36.72, "lpips": 0.019, "jod": 7.62},
    "aces": {"kl_sdr": 2.983, "pu21_psnr": 39.30, "lpips": 0.016, "jod": 7.40},
}

# Table 5 — augmentation ablation
TABLE5_AUGMENTATION: dict[str, dict[str, float]] = {
    "full": {"pu21_psnr": 36.97, "jod": 7.86},
    "no_aug": {"pu21_psnr": 39.00, "jod": 7.43},
    "blur_only": {"pu21_psnr": 33.12, "jod": 6.90},
}

TRAINING_DEFAULTS = {
    "steps": 10_000,
    "clips_approx": 300,
    "lora_param_fraction": 0.01,
    "denoise_steps_inference": 11,
    "vae_encoding": "logc3",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_vae_roundtrip": TABLE1_VAE_ROUNDTRIP,
        "table2_baselines_arri": TABLE2_BASELINES_ARRI,
        "table2_baselines_upiq": TABLE2_BASELINES_UPIQ,
        "table3_temporal": TABLE3_TEMPORAL,
        "table4_encoding": TABLE4_ENCODING,
        "table5_augmentation": TABLE5_AUGMENTATION,
        "training_defaults": TRAINING_DEFAULTS,
    }
