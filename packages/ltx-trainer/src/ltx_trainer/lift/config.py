"""LiFT — Lifted inter-slice Feature Trajectories (arXiv:2605.19060)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LiftConfig:
    paper_arxiv: str = "2605.19060"
    paper_title: str = (
        "LiFT: Lifted Inter-slice Feature Trajectories for 3D Image Generation from 2D Generators"
    )

    # Variants
    variant_unconditional: str = "LiFT-U"
    variant_conditional: str = "LiFT-C"

    # LiFT-U (BraTS unconditional)
    lift_u_mapper_params_m: float = 1.455
    lift_u_drift_backbone: str = "ResNet-18 (ImageNet, frozen)"
    lift_u_best_fid_x1e3: float = 0.066
    lift_u_ms_ssim: float = 0.543
    lift_u_inference_mem_gb: float = 0.41
    lift_u_resolution: tuple[int, int, int] = (128, 128, 128)

    # LiFT-C missing-MR (BraTS)
    missing_mr_resolution: tuple[int, int, int] = (240, 240, 155)
    missing_mr_validation_n: int = 219
    lift_c_inference_s: float = 1.16
    cwdm_inference_s: float = 156.40
    speedup_vs_cwdm: float = 135.0  # paper-reported ratio on RTX 5090 setup

    # LiFT-C MR-to-CT (SynthRAD2023)
    synthrad_train: int = 144
    synthrad_test: int = 36
    mrct_resolution: tuple[int, int, int] = (128, 128, 128)
    lift_c_mae_hu: float = 57.50
    lift_c_psnr: float = 28.49
    lift_c_ssim: float = 0.8740
    lift_c_ncc: float = 0.9200
    lift_c_dz_mae_full: float = 38.41
    lift_c_dz_corr: float = 0.729

    views_six: tuple[str, ...] = ("R1", "L1", "R2", "L2", "R3", "L3")
    tri_planes: tuple[str, ...] = ("axial", "coronal", "sagittal")

    classifiers_evaluated: tuple[str, ...] = (
        "Decision-Tree",
        "Random-Forest",
        "SVM",
        "MLP",
        "MLP-Large",
        "TabPFN",
    )

    datasets: tuple[str, ...] = (
        "BraTS 2023 GLI (Tasks A & B)",
        "SynthRAD2023 Task 1 Brain (Task C)",
    )
