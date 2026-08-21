"""Paper result tables for NAS-VAR (arXiv:2605.19354)."""

from __future__ import annotations

from typing import Any


def table1_cartesian_x() -> list[dict[str, Any]]:
    """Table 1 — fastMRI ES Cartesian-X, R=32 (selected methods)."""
    return [
        {"method": "MambaRecon", "PSNR_T1": 24.09, "PSNR_T2": 19.63, "PSNR_FLAIR": 18.53, "LPIPS_T1": 0.30},
        {
            "method": "Proposed",
            "PSNR_T1": 24.61,
            "PSNR_T2": 19.16,
            "PSNR_FLAIR": 21.29,
            "LPIPS_T1": 0.16,
            "LPIPS_T2": 0.24,
            "LPIPS_FLAIR": 0.22,
        },
    ]


def table2_radial() -> list[dict[str, Any]]:
    """Table 2 — fastMRI Radial, R=32."""
    return [
        {"method": "MambaRecon", "PSNR_T1": 27.16, "LPIPS_T1": 0.23},
        {"method": "Proposed", "PSNR_T1": 26.00, "PSNR_T2": 20.51, "PSNR_FLAIR": 22.68, "LPIPS_T1": 0.16},
    ]


def table3_cartesian_y() -> list[dict[str, Any]]:
    """Table 3 — fastMRI ES Cartesian-Y, R=32."""
    return [
        {"method": "MambaRecon", "PSNR_FLAIR": 18.15},
        {"method": "Proposed", "PSNR_T1": 24.11, "PSNR_FLAIR": 20.96, "LPIPS_T1": 0.17},
    ]


def table5_distillation_cartesian_x() -> list[dict[str, Any]]:
    """Table 5 excerpt — distillation on ES Cartesian-X."""
    return [
        {"model": "Base", "PSNR_T1": 24.14, "PSNR_FLAIR": 21.07, "SSIM_FLAIR": 0.57},
        {"model": "Distilled", "PSNR_T1": 24.61, "PSNR_FLAIR": 21.29, "SSIM_FLAIR": 0.60},
    ]


def table4_ablations_summary() -> list[dict[str, Any]]:
    """Table 4 — ablation variants (avg over contrasts/masks, argmax)."""
    return [
        {"variant": "w/o Cross-Attention", "note": "Lower PSNR/SSIM, higher LPIPS"},
        {"variant": "w/o Token Hierarchy", "note": "Flat 16x16 at all scales hurts badly"},
        {"variant": "w/o Trainable Encoder", "note": "Cross-attn context degraded"},
        {"variant": "Base (Argmax Decoding)", "note": "Best overall; beats multinomial/top-p"},
    ]


def supp_table2_mean_perceptual() -> list[dict[str, Any]]:
    """Supp. Table 2 — mean Alex/VGG/DISTS (Proposed best on all masks)."""
    return [
        {"mask": "Cartesian-X", "Alex_LPIPS": 0.21, "VGG_LPIPS": 0.30, "DISTS": 0.19},
        {"mask": "Cartesian-Y", "Alex_LPIPS": 0.20, "VGG_LPIPS": 0.31, "DISTS": 0.18},
        {"mask": "Gaussian-VD", "Alex_LPIPS": 0.16, "VGG_LPIPS": 0.26, "DISTS": 0.16},
        {"mask": "Radial", "Alex_LPIPS": 0.18, "VGG_LPIPS": 0.28, "DISTS": 0.17},
    ]
