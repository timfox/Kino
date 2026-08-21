"""Paper benchmark tables (Tables 1–9)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.feformer.config import FLOPS_G, PARAMS_M


def table1_datasets() -> list[dict[str, Any]]:
    return [
        {"dataset": "AMOS 2022", "task": "Multi-Organs", "modality": "CT", "size": 300, "classes": 15},
        {"dataset": "Hepatic Vessel", "task": "Tubular&Tumor", "modality": "CT", "size": 303, "classes": 2},
        {"dataset": "Brain Tumor", "task": "Tissues", "modality": "Multi-modal MR", "size": 484, "classes": 3},
        {"dataset": "FLARE", "task": "Abdominal Organs", "modality": "CT", "size": 361, "classes": 4},
    ]


def table2_amos_mean() -> dict[str, float]:
    return {"feformer_mean_dsc": 90.11, "feformer_mean_hd95_mm": 1.78, "nnunet_mean_dsc": 88.21}


def table3_hepatic_flare() -> dict[str, dict[str, float]]:
    return {
        "hepatic_vessel_tumor": {"feformer_dsc": 67.97, "feformer_hd95": 9.94},
        "flare": {"feformer_dsc": 95.02, "feformer_hd95": 1.40},
    }


def table4_brain_tumor() -> dict[str, float]:
    return {"feformer_mean_dsc": 74.97, "feformer_mean_hd95": 5.01}


def table5_complexity() -> dict[str, Any]:
    return {
        "feformer": {"params_m": PARAMS_M, "flops_g": FLOPS_G},
        "nnunet": {"params_m": 68.38, "flops_g": 357.13},
        "nnformer": {"params_m": 149.33, "flops_g": 284.28},
        "vsmtrans": {"params_m": 50.39, "flops_g": 358.21},
    }


def table6_ablation_modules() -> list[dict[str, Any]]:
    return [
        {"fdsa": False, "fgmlp": False, "waff": False, "fcsb": False, "mean_dsc": 84.08, "mean_hd95": 2.86},
        {"fdsa": True, "fgmlp": False, "waff": False, "fcsb": False, "mean_dsc": 86.32, "mean_hd95": 2.16},
        {"fdsa": False, "fgmlp": True, "waff": False, "fcsb": False, "mean_dsc": 86.21, "mean_hd95": 2.21},
        {"fdsa": True, "fgmlp": True, "waff": False, "fcsb": False, "mean_dsc": 87.56, "mean_hd95": 1.95},
        {"fdsa": True, "fgmlp": True, "waff": True, "fcsb": False, "mean_dsc": 88.98, "mean_hd95": 1.83},
        {"fdsa": True, "fgmlp": True, "waff": True, "fcsb": True, "mean_dsc": 90.11, "mean_hd95": 1.78},
    ]


def table7_fdsa_fgmlp_ablation() -> list[dict[str, Any]]:
    return [
        {"module": "FDSA", "config": "standard self-attention", "mean_dsc": 87.85, "mean_hd95": 2.07},
        {"module": "FDSA", "config": "frequency-domain self-attention", "mean_dsc": 89.06, "mean_hd95": 1.85},
        {"module": "FDSA", "config": "+ multi-frequency dynamic", "mean_dsc": 90.11, "mean_hd95": 1.78},
        {"module": "FGMLP", "config": "standard MLP", "mean_dsc": 88.01, "mean_hd95": 2.02},
        {"module": "FGMLP", "config": "gating mechanism", "mean_dsc": 88.75, "mean_hd95": 1.90},
        {"module": "FGMLP", "config": "+ selective frequency decomposition", "mean_dsc": 90.11, "mean_hd95": 1.78},
    ]


def table8_fusion_comparison() -> list[dict[str, Any]]:
    return [
        {"module": "AFF", "mean_dsc": 88.82, "mean_hd95": 1.89, "params_m": 18.63},
        {"module": "DFF", "mean_dsc": 89.54, "mean_hd95": 1.81, "params_m": 18.97},
        {"module": "WAFF", "mean_dsc": 90.11, "mean_hd95": 1.78, "params_m": 18.54},
    ]


def table9_flare_generalization() -> dict[str, Any]:
    return {
        "feformer": {"internal_dsc": 95.02, "external_dsc": 88.54, "dsc_gap": 6.48},
        "nnunet": {"internal_dsc": 94.30, "external_dsc": 85.45, "dsc_gap": 8.85},
        "unetr": {"internal_dsc": 92.54, "external_dsc": 80.65, "dsc_gap": 11.89},
    }
