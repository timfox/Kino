"""Paper Table I–II excerpts (ST-SFLora, arXiv:2605.26120)."""

from __future__ import annotations

from typing import Any


def table1_top1_accuracy() -> dict[str, Any]:
    """Table I — Top-1 accuracy (%) excerpts (ViT-B/16 Non-IID column highlighted)."""
    return {
        "note": "Full table spans ViT-S/B/L × IID/Non-IID × 6 methods × 3 datasets.",
        "vit_b16_non_iid": {
            "ImageNet100": {
                "LocalLoRA": 45.64,
                "FedLoRA": 52.49,
                "SplitLoRA": 89.47,
                "SFLora": 89.09,
                "ST-SFLora-Full": 89.14,
                "ST-SFLora": 85.81,
            },
            "Oxford_Flowers-102": {
                "LocalLoRA": 75.43,
                "FedLoRA": 57.95,
                "SplitLoRA": 99.29,
                "SFLora": 98.69,
                "ST-SFLora-Full": 98.74,
                "ST-SFLora": 96.43,
            },
            "CUB-200-2011": {
                "LocalLoRA": 38.97,
                "FedLoRA": 21.53,
                "SplitLoRA": 80.65,
                "SFLora": 79.84,
                "ST-SFLora-Full": 79.91,
                "ST-SFLora": 73.69,
            },
        },
    }


def table2_client_overhead() -> dict[str, Any]:
    """Table II — client-side computation and communication (ViT-B/16)."""
    return {
        "columns": ["gpu_mem_gb", "model_broadcast_mb", "lora_mb", "token_activation_mb"],
        "rows": {
            "LocalLoRA": [9.0, 335.3, 7.9, 0.0],
            "FedLoRA": [9.0, 335.3, 7.9, 0.0],
            "SplitLoRA": [2.3, 0.0, 1.3, 3.0 * 196 / 16],
            "SFLora": [2.3, 0.0, 1.3, 3.0 * 196 / 16],
            "ST-SFLora-Full": [1.4, 0.0, 1.3, 3.0 * 196 / 16],
            "ST-SFLora_topK": [1.4, 0.0, 1.3, "3K/16 (K adaptive)"],
        },
        "token_mb_formula": "3/16 = B·D·q0 / 1024^2 with B=64, D=768, q0=32",
    }


def headline_results() -> dict[str, Any]:
    return {
        "lowest_client_gpu_mem_gb": 1.4,
        "lowest_client_comm_vs_splitlora": "top-K tokens vs full N=196 patches",
        "ste_joint_optimization": "Power + bandwidth + token budget alternating (Alg. 4)",
        "vit_b16_non_iid_st_vs_splitlora_imagenet100_pts": 85.81,
        "vit_b16_non_iid_splitlora_imagenet100_pts": 89.47,
    }


def fig8_ste_ablation_labels() -> dict[str, str]:
    return {
        "full": "Joint optimization (power + bandwidth + token selection)",
        "no_power": "Fixed power — moderate STE drop",
        "no_bandwidth": "Fixed bandwidth — larger STE drop",
        "no_token_selection": "Full-token uplink — largest STE drop",
    }
