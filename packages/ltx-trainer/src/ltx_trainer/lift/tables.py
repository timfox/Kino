"""Paper tables for LiFT (arXiv:2605.19060)."""

from __future__ import annotations

from typing import Any


def table1_unconditional_fid() -> list[dict[str, Any]]:
    """Table 1 — BraTS GLI unconditional 128³ (FID ×10³, MS-SSIM, inference mem)."""
    return [
        {"method": "2.5D LDM", "fid_x1e3": 81.06, "ms_ssim": 0.579, "mem_gb": 6.81},
        {"method": "3D DDPM", "fid_x1e3": 1.402, "ms_ssim": 0.876, "mem_gb": 6.51},
        {"method": "3D LDM", "fid_x1e3": 1.394, "ms_ssim": 0.926, "mem_gb": 9.82},
        {"method": "HA-GAN", "fid_x1e3": 0.785, "ms_ssim": 0.905, "mem_gb": 2.58},
        {"method": "WDM (WavU-Net)", "fid_x1e3": 0.259, "ms_ssim": 0.879, "mem_gb": 2.65},
        {"method": "WDM", "fid_x1e3": 0.154, "ms_ssim": 0.888, "mem_gb": 2.55},
        {"method": "LiFT-U", "fid_x1e3": 0.066, "ms_ssim": 0.543, "mem_gb": 0.41},
    ]


def table1_mlp_per_view() -> list[dict[str, Any]]:
    """Table 1 excerpt — MLP F1 per view (Day1–2 difference, LiFT-U ablation context)."""
    return [
        {"view": "Left-3", "f1": 0.68, "ci": "0.48–0.84"},
        {"view": "Right-3", "f1": 0.61, "ci": "0.40–0.82"},
        {"view": "All Views", "f1": 0.80, "ci": "0.62–0.96"},
    ]


def fusion_heatmap_mlp() -> list[dict[str, Any]]:
    """Fig. 3 — fusion × temporal (MLP, All Views)."""
    return [
        {"fusion": "Concatenate", "temporal_concat": 0.57, "temporal_difference": 0.80},
        {"fusion": "Avg. Features", "temporal_concat": 0.45, "temporal_difference": 0.76},
        {"fusion": "Max. Votes", "temporal_concat": 0.55, "temporal_difference": 0.61},
    ]


def table2_missing_mr() -> list[dict[str, Any]]:
    """Table 2 — missing-MR PSNR/SSIM per contrast (N=219 validation)."""
    return [
        {
            "method": "pix2pix",
            "T1n": "27.61 / 0.9475",
            "T1c": "25.91 / 0.9316",
            "T2w": "26.69 / 0.9419",
            "T2f": "25.19 / 0.9169",
            "inference_s": 0.05,
        },
        {
            "method": "cWDM",
            "T1n": "29.74 / 0.9622",
            "T1c": "27.32 / 0.9451",
            "T2w": "28.81 / 0.9588",
            "T2f": "27.83 / 0.9438",
            "inference_s": 156.40,
        },
        {
            "method": "LiFT-C, no mapper",
            "T1n": "28.97 / 0.9608",
            "T1c": "27.30 / 0.9455",
            "T2w": "28.40 / 0.9567",
            "T2f": "26.46 / 0.9393",
            "inference_s": None,
        },
        {
            "method": "LiFT-C",
            "T1n": "29.42 / 0.9615",
            "T1c": "27.40 / 0.9460",
            "T2w": "28.53 / 0.9571",
            "T2f": "27.88 / 0.9424",
            "inference_s": 1.16,
        },
    ]


def table3_mr_to_ct() -> list[dict[str, Any]]:
    """Table 3 — SynthRAD2023 MR-to-CT (n=36 test)."""
    return [
        {"method": "Pix2pix-UNet", "mae_hu": 78.38, "psnr": 26.41, "ssim": 0.8082, "ncc": 0.8700},
        {"method": "Pix2pix-ResNet", "mae_hu": 59.30, "psnr": 28.43, "ssim": 0.8637, "ncc": 0.9172},
        {"method": "CBAM3D-UNet", "mae_hu": 64.52, "psnr": 27.82, "ssim": 0.8500, "ncc": 0.9073},
        {"method": "LiFT-C, no mapper", "mae_hu": 59.40, "psnr": 28.31, "ssim": 0.8679, "ncc": 0.9163},
        {"method": "LiFT-C", "mae_hu": 57.50, "psnr": 28.49, "ssim": 0.8740, "ncc": 0.9200},
    ]


def table4_through_plane_mrct() -> list[dict[str, Any]]:
    """Table 4 — through-plane ∆z coherence MR-to-CT."""
    return [
        {"method": "Pix2pix-UNet", "dz_mae_full": 52.20, "dz_mae_bone": 180.23, "dz_corr": 0.575},
        {"method": "Pix2pix-ResNet", "dz_mae_full": 44.91, "dz_mae_bone": 159.54, "dz_corr": 0.650},
        {"method": "CBAM3D-UNet", "dz_mae_full": 39.88, "dz_mae_bone": 141.10, "dz_corr": 0.713},
        {"method": "LiFT-C, no mapper", "dz_mae_full": 42.73, "dz_mae_bone": 151.73, "dz_corr": 0.675},
        {"method": "LiFT-C", "dz_mae_full": 38.41, "dz_mae_bone": 135.52, "dz_corr": 0.729},
    ]


def training_hyperparameters() -> list[dict[str, Any]]:
    """Table 10 — main training settings excerpt."""
    return [
        {"task": "LiFT-U Stage 1", "module": "2D axial generator", "optimizer": "Adam 2e-4", "batch": "128 slices"},
        {"task": "LiFT-U Stage 2", "module": "depth mapper", "optimizer": "Adam 2e-4", "batch": "8 volumes"},
        {"task": "Missing-MR LiFT-C", "module": "2D U-Net + BiGRU", "optimizer": "AdamW 2e-5/2e-4", "batch": "1 vol, 48 slices"},
        {"task": "MR-to-CT Stage 1", "module": "2D U-Net", "optimizer": "Adam 2e-4", "batch": "32 slices"},
        {"task": "MR-to-CT Stage 2", "module": "BiGRU residual", "optimizer": "AdamW 2e-4", "batch": "32 windows"},
    ]
