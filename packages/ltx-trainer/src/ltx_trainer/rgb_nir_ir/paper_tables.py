"""Table 1 and paper metric excerpts."""

from __future__ import annotations

from typing import Any


def table_i_inverse_rendering() -> list[dict[str, Any]]:
    """Table 1 — synthetic + real aggregate (paper)."""
    return [
        {
            "method": "NeILF",
            "albedo_psnr": 16.61,
            "albedo_ssim": 0.8318,
            "albedo_lpips": 0.1409,
            "roughness_rmse": 0.4138,
            "normal_mae": 20.70,
            "relighting_psnr": None,
            "hours": 12.0,
        },
        {
            "method": "TensoIR",
            "albedo_psnr": 17.40,
            "albedo_ssim": 0.9057,
            "albedo_lpips": 0.0875,
            "roughness_rmse": 0.1586,
            "normal_mae": 19.30,
            "relighting_psnr": 20.22,
            "hours": 6.0,
        },
        {
            "method": "GS-IR",
            "albedo_psnr": 17.24,
            "albedo_ssim": 0.8812,
            "albedo_lpips": 0.1277,
            "roughness_rmse": 0.2207,
            "normal_mae": 16.37,
            "relighting_psnr": 27.17,
            "hours": 0.7,
        },
        {
            "method": "R3DG",
            "albedo_psnr": 16.69,
            "albedo_ssim": 0.8871,
            "albedo_lpips": 0.1173,
            "roughness_rmse": 0.1371,
            "normal_mae": 13.35,
            "relighting_psnr": 26.11,
            "hours": 1.0,
        },
        {
            "method": "IRGS",
            "albedo_psnr": 13.96,
            "albedo_ssim": 0.8580,
            "albedo_lpips": 0.0827,
            "roughness_rmse": 0.1174,
            "normal_mae": 10.48,
            "relighting_psnr": 26.45,
            "hours": 2.0,
        },
        {
            "method": "WildLight",
            "albedo_psnr": 21.10,
            "albedo_ssim": 0.9328,
            "albedo_lpips": 0.1053,
            "roughness_rmse": 0.1472,
            "normal_mae": 7.84,
            "relighting_psnr": None,
            "hours": 6.0,
        },
        {
            "method": "Ours",
            "albedo_psnr": 27.89,
            "albedo_ssim": 0.9675,
            "albedo_lpips": 0.0680,
            "roughness_rmse": 0.0713,
            "normal_mae": 7.73,
            "relighting_psnr": 31.01,
            "hours": 1.2,
        },
    ]
