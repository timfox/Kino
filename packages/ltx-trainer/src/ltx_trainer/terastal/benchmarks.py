"""Paper tables and evaluation anchors (Tables I–II, Figs 4–6)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.terastal.constants import (
    AVG_NORMALIZED_ACCURACY_LOSS,
    HARDWARE_SETTINGS,
    MISS_RATE_REDUCTION_VS,
)


def table_i_hardware() -> list[dict[str, Any]]:
    return [
        {
            "total_pes": 4000,
            "partition": "1 WS (2K) + 2 OS (1K each)",
            "scenarios": ["AR social interaction", "AR gaming (light)", "Multi-Camera Vision (light)"],
        },
        {
            "total_pes": 6000,
            "partition": "1 WS (2K) + 2 OS (2K each)",
            "scenarios": ["AR social interaction", "AR gaming (heavy)", "Multi-Camera Vision (heavy)"],
        },
    ]


def table_ii_workloads() -> list[dict[str, Any]]:
    return [
        {
            "scenario": "AR Social Interaction",
            "models": [
                {"name": "FBNet-C", "fps": 60, "variants": False},
                {"name": "Hand S/P", "fps": 30, "variants": False},
                {"name": "Sp2Dense", "fps": 30, "variants": True},
                {"name": "MobileNetV2-SSD", "fps": 30, "variants": True},
            ],
        },
        {
            "scenario": "Multi-Camera Vision (Light)",
            "models": [
                {"name": "MobileNetV2-SSD", "fps": 45, "variants": True},
                {"name": "ResNet50", "fps": 15, "variants": True},
                {"name": "VGG11", "fps": 15, "variants": True},
                {"name": "InceptionV3", "fps": 15, "variants": True},
                {"name": "Swin-Tiny", "fps": 10, "variants": True},
            ],
        },
        {
            "scenario": "Multi-Camera Vision (Heavy)",
            "models": [
                {"name": "MobileNetV2-SSD", "fps": 60, "variants": True},
                {"name": "ResNet50", "fps": 30, "variants": True},
                {"name": "VGG11", "fps": 30, "variants": True},
                {"name": "InceptionV3", "fps": 15, "variants": True},
                {"name": "Swin-Tiny", "fps": 30, "variants": True},
            ],
        },
    ]


def related_work_positioning() -> list[dict[str, bool | str]]:
    return [
        {"study": "DREAM [1]", "heterogeneity": True, "layer_variants": False, "virtual_budgets": False},
        {"study": "DARIS [9]", "heterogeneity": False, "layer_variants": False, "virtual_budgets": True},
        {"study": "BlastNet [17]", "heterogeneity": False, "layer_variants": True, "virtual_budgets": False},
        {"study": "Terastal (present)", "heterogeneity": True, "layer_variants": True, "virtual_budgets": True},
    ]


def miss_rate_comparison_anchors() -> dict[str, Any]:
    """Fig. 5 summary: Terastal lowest mean miss rate across hardware settings."""
    return {
        "reduction_vs_fcfs_pct": round(MISS_RATE_REDUCTION_VS["fcfs"] * 100, 2),
        "reduction_vs_edf_pct": round(MISS_RATE_REDUCTION_VS["edf"] * 100, 2),
        "reduction_vs_dream_pct": round(MISS_RATE_REDUCTION_VS["dream"] * 100, 2),
        "avg_normalized_accuracy_loss_pct": round(AVG_NORMALIZED_ACCURACY_LOSS * 100, 2),
        "ablations": {
            "no_variants": "virtual budgets alone beat FCFS/EDF/DREAM",
            "no_budgeting": "variants without budgets weaker than full Terastal",
        },
    }


def storage_overhead_range() -> dict[str, float]:
    return {"min_pct": 0.5, "max_pct": 5.9}


def summary_anchors() -> dict[str, Any]:
    return {
        "miss_rate_reduction": MISS_RATE_REDUCTION_VS,
        "accuracy_loss": AVG_NORMALIZED_ACCURACY_LOSS,
        "hardware": HARDWARE_SETTINGS,
        "simulator": ("MAESTRO", "XRBench"),
        "storage_overhead_pct": storage_overhead_range(),
    }
