"""Paper anchors — ErA defocus deblurring (Vo & Park, arXiv:2606.06540)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06540"
PAPER_TITLE = "ErA: Error-Aware Deep Unrolling Network for Single Image Defocus Deblurring"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "NeurIPS 2025 COML Workshop"

UNROLLING_DEPTH_K = 10
KERNEL_SIZE = 61
PATCH_SIZE = 140
TRAIN_EPOCHS = 150

DATASET_DPDD = {
    "name": "DPDD",
    "train": 350,
    "val": 74,
    "test": 76,
    "format": "16-bit dual-pixel pairs",
}

# Table 1 — PSNR on DPDD / RealDOF / RTF
TABLE1_RESULTS: tuple[dict[str, str | float], ...] = (
    {
        "method": "DRBNet",
        "DPDD_PSNR": 25.485,
        "DPDD_SSIM": 0.792,
        "RealDOF_PSNR": 24.700,
        "RTF_PSNR": 24.463,
    },
    {
        "method": "NRKNet",
        "DPDD_PSNR": 26.110,
        "DPDD_SSIM": 0.810,
        "RealDOF_PSNR": 25.060,
        "RTF_PSNR": 25.931,
    },
    {
        "method": "P2IKT",
        "DPDD_PSNR": 26.280,
        "DPDD_SSIM": 0.807,
        "RealDOF_PSNR": 25.480,
        "RTF_PSNR": 25.260,
    },
    {
        "method": "IRNeXt",
        "DPDD_PSNR": 26.300,
        "DPDD_SSIM": 0.814,
        "RealDOF_PSNR": 25.660,
        "RTF_PSNR": 25.333,
    },
    {
        "method": "ErA w/o E",
        "DPDD_PSNR": 26.361,
        "DPDD_SSIM": 0.812,
        "RealDOF_PSNR": 25.422,
        "RTF_PSNR": 25.084,
    },
    {
        "method": "ErA",
        "DPDD_PSNR": 26.687,
        "DPDD_SSIM": 0.815,
        "RealDOF_PSNR": 25.747,
        "RTF_PSNR": 25.502,
    },
)

ERA_ROW = next(r for r in TABLE1_RESULTS if r["method"] == "ErA")
ERA_WO_E_ROW = next(r for r in TABLE1_RESULTS if r["method"] == "ErA w/o E")
BEST_BASELINE_PSNR = max(
    float(r["DPDD_PSNR"]) for r in TABLE1_RESULTS if r["method"] not in ("ErA", "ErA w/o E")
)

TRAIN_DEFAULTS = {
    "optimizer": "Adam",
    "lr": 1e-4,
    "epochs": TRAIN_EPOCHS,
    "patch_size": PATCH_SIZE,
    "unrolling_depth": UNROLLING_DEPTH_K,
    "kernel_size": KERNEL_SIZE,
    "omega_recon": 0.5,
}
