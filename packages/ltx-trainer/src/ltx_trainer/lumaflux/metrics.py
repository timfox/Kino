"""Paper benchmark metrics (Tables 1–4)."""

from __future__ import annotations

TABLE1_BENCHMARKS: dict[str, dict[str, dict[str, float]]] = {
    "hdrtv1k": {
        "lumaflux": {"psnr": 39.27, "ssim": 0.982, "hdr_vdp3": 9.83, "delta_e_itp": 6.12},
        "hdrtvnet++": {"psnr": 38.36, "ssim": 0.973, "hdr_vdp3": 8.75, "delta_e_itp": 8.28},
        "lediff": {"psnr": 36.52, "ssim": 0.872, "hdr_vdp3": 5.71, "delta_e_itp": 9.13},
    },
    "hdrtv4k": {
        "lumaflux": {"psnr": 35.86, "ssim": 0.978, "hdr_vdp3": 9.72, "delta_e_itp": 5.86},
        "hdrtvnet++": {"psnr": 30.82, "ssim": 0.881, "hdr_vdp3": 8.12, "delta_e_itp": 7.85},
    },
    "luma_eval": {
        "lumaflux": {"psnr": 36.92, "ssim": 0.938, "hdr_vdp3": 8.91, "delta_e_itp": 5.67},
        "hdrtvnet++": {"psnr": 36.54, "ssim": 0.901, "hdr_vdp3": 8.22, "delta_e_itp": 7.35},
    },
}

TABLE2_TMO: list[dict[str, float | str]] = [
    {"tmo": "OCIOv2", "psnr": 37.85, "psnr_y": 38.94, "delta_e_itp": 5.41, "hdr_vdp3": 9.12},
    {"tmo": "2446c+GM", "psnr": 38.31, "psnr_y": 39.12, "delta_e_itp": 5.18, "hdr_vdp3": 9.27},
    {"tmo": "Expert Graded SDR", "psnr": 37.11, "psnr_y": 38.02, "delta_e_itp": 5.53, "hdr_vdp3": 9.03},
    {"tmo": "Reinhard", "psnr": 35.88, "psnr_y": 36.42, "delta_e_itp": 6.45, "hdr_vdp3": 8.31},
]

TABLE3_ABLATION: list[dict[str, float | str]] = [
    {"variant": "Flux + LoRA only", "psnr": 33.42, "delta_e_itp": 8.58, "fr_hidrovqa": 72.9},
    {"variant": "+ PGA (no spectral)", "psnr": 34.94, "delta_e_itp": 7.62, "fr_hidrovqa": 75.2},
    {"variant": "+ PGA (with spectral gating)", "psnr": 35.18, "delta_e_itp": 7.31, "fr_hidrovqa": 76.3},
    {"variant": "+ PCM (SigLIP FiLM)", "psnr": 35.89, "delta_e_itp": 6.78, "fr_hidrovqa": 78.6},
    {"variant": "+ RQS (monotone spline)", "psnr": 35.98, "delta_e_itp": 6.09, "fr_hidrovqa": 80.8},
]

TABLE4_USER_STUDY: dict[str, dict[str, float]] = {
    "lumaflux": {"brightness": 3.8, "color": 4.5, "overall": 4.2},
    "lediff": {"brightness": 3.1, "color": 4.0, "overall": 4.0},
    "hdrtvnet++": {"brightness": 3.5, "color": 3.6, "overall": 3.5},
}
