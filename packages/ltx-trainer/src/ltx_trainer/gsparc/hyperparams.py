"""Table 2 training hyperparameters."""

from __future__ import annotations

from typing import Any


def training_hyperparameters() -> dict[str, Any]:
    return {
        "lambda_l1_ssim": 0.2,
        "position_lr_init": 1.6e-4,
        "position_lr_final": 1.6e-7,
        "opacity_lr": 0.0055,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "emission_mlp_lr": 0.005,
        "emission_mlp_arch": "3 → 16 → 2",
        "confidence_alpha": 0.3,
        "confidence_lr": 0.005,
        "confidence_mlp_arch": "3 → 32 → 1",
        "optimizer": "Adam",
        "hardware_note": "NVIDIA GeForce RTX 2080 Ti (paper)",
    }
