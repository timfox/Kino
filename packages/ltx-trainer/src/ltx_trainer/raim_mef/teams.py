"""Top-team method summaries from challenge report §4."""

from __future__ import annotations

from typing import Any

TEAM_METHODS: dict[str, dict[str, Any]] = {
    "WHU-VIP": {
        "rank": 1,
        "backbone": "AFUNet (Swin) + SFT + DCM-RG reliability gating",
        "inputs": "3 frames: 0 EV, ±2 EV",
        "training": "4-stage curriculum + EMA; score-aligned loss",
        "final_score": 58.889,
    },
    "SHL": {
        "rank": 2,
        "backbone": "Ensemble Restormer / Uformer / MST++",
        "inputs": "5 of 7 exposures + downsampled global guidance",
        "training": "Charbonnier + SSIM + gradient + NPR loss; 60k iters",
        "final_score": 58.034,
    },
    "nunucccb": {
        "rank": 3,
        "backbone": "HDR-Transformer (CA-ViT) + spatial attention alignment",
        "inputs": "5 LDR frames, f3 reference",
        "training": "L1 + VGG perceptual; 150k iters, 256² crops",
        "final_score": 57.000,
    },
    "untrafusion": {
        "rank": 4,
        "backbone": "SPyNet flow + Restormer U-Net + exposure alignment",
        "inputs": "variable frame count (7 train / 5 test)",
        "training": "progressive patch 256→512→768; TLC inference",
        "final_score": 56.033,
    },
    "I2_Group_Transsion": {
        "rank": 5,
        "backbone": "Haar DWT + DehazeFormer-s (LL) + HF residual fusion",
        "inputs": "5 exposures → 15 channels",
        "training": "wavelet branch decoupling",
        "final_score": 55.236,
    },
    "NTR": {
        "rank": None,
        "backbone": "TimeDiffiT / time-conditioned U-Net (15-ch)",
        "inputs": "5 evenly spaced exposures",
        "training": "metric-aligned L1+SSIM+LPIPS; 8-way TTA",
        "final_score": None,
    },
}


def team_cards() -> dict[str, Any]:
    return TEAM_METHODS
