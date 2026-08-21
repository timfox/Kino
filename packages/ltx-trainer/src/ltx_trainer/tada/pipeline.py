"""TADA framework card, paper tables, and smoke demos (arXiv:2605.21523)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.tada.config import TADAConfig
from ltx_trainer.tada.emulator import (
    DENOISE_KERNEL_TADA,
    SHARPEN_KERNEL_TADA,
    apply_emulator,
    project_to_symmetric_sum_one,
)
from ltx_trainer.tada.layout import LIMITATIONS
from ltx_trainer.tada.loss import regret_stub, tada_loss
from ltx_trainer.tada.mock import toy_target_pair
from ltx_trainer.tada.residuals import kb_residual


def framework_card(cfg: TADAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TADAConfig()
    return {
        "name": "TADA",
        "paper": cfg.paper_arxiv,
        "doi": cfg.paper_doi,
        "github": cfg.github,
        "idea": (
            "Target Alignment through Data Adaptation: learn a lightweight convolutional emulator "
            "so residual statistics of an emulated JPEG source match an unlabeled operational target, "
            "mitigating Cover Source Mismatch (CSM) in JPEG steganalysis."
        ),
        "components": [
            "amaze demosaick (fixed)",
            "symmetric-sum-to-1 convolution emulator",
            "differentiable JPEG (external in paper)",
            "KB residual extractor E",
            "loss: covariance Frobenius + W1 distribution + ℓ2 realism",
        ],
        "operational_assumptions": [
            "homogeneous target pipeline ω_t",
            "known embedding γ_t (Kerckhoffs)",
            "shared JPEG quantization table",
            "small unlabeled operational set",
            "unknown cover/stego balance",
        ],
        "defaults": cfg.__dict__,
    }


def table_i_toy_kernels_detailed() -> dict[str, Any]:
    """Table 1 — toy denoising/sharpening kernels and regrets (%)."""
    return {
        "denoising": {
            "original": [
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625],
            ],
            "tada_learned": DENOISE_KERNEL_TADA.tolist(),
            "regret_naive_pct": 50,
            "regret_tada_pct": 2,
        },
        "sharpening": {
            "original": [
                [0.0, -0.25, 0.0],
                [-0.25, 2.0, -0.25],
                [0.0, -0.25, 0.0],
            ],
            "tada_learned": SHARPEN_KERNEL_TADA.tolist(),
            "regret_naive_pct": 27,
            "regret_tada_pct": 0,
        },
    }


def table_ii_flickr_targets() -> list[dict[str, str | int]]:
    """Table 2 — YFCC100M operational targets."""
    return [
        {"target": "SONY", "camera": "SONY SLT A37", "jpeg_quality_factor": 90},
        {"target": "NIKON", "camera": "NIKON D40", "jpeg_quality_factor": 90},
        {"target": "CANON", "camera": "Canon PowerShot SX30 IS", "jpeg_quality_factor": 93},
    ]


def table_iii_target_regrets() -> list[dict[str, Any]]:
    """Table 3 — target regrets (%) for baselines and TADA operational regimes."""
    return [
        {
            "target": "SONY",
            "intrinsic_difficulty_pct": 1,
            "naive": 37,
            "chordal_min": 27,
            "all_holistic": 14,
            "multiclassifier": 27,
            "tada_full_cover": 7,
            "tada_full_stego": 9,
            "tada_mix": 8,
        },
        {
            "target": "NIKON",
            "intrinsic_difficulty_pct": 3,
            "naive": 30,
            "chordal_min": 17,
            "all_holistic": 7,
            "multiclassifier": 0,
            "tada_full_cover": 9,
            "tada_full_stego": 6,
            "tada_mix": 7,
        },
        {
            "target": "CANON",
            "intrinsic_difficulty_pct": 1,
            "naive": 45,
            "chordal_min": 48,
            "all_holistic": 36,
            "multiclassifier": 35,
            "tada_full_cover": 27,
            "tada_full_stego": 24,
            "tada_mix": 26,
        },
    ]


def csm_formalization() -> dict[str, str]:
    """Sec. 2.1 — intrinsic difficulty and regret definitions (text)."""
    return {
        "intrinsic_difficulty": "P_E(f(x|θ_ω,γ) ≠ y) when train and test share target pipeline (ω,γ)",
        "regret": "R(S,T) = P_E(f(x|θ_S) ≠ y on T) − intrinsic_difficulty(T)",
        "objective": "S* = argmin_S R(S,T) via emulated source S_TADA aligned to target residuals",
    }


def pipeline_demo(cfg: TADAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TADAConfig()
    raw, target = toy_target_pair("denoise", seed=3)
    emulated = apply_emulator(raw, DENOISE_KERNEL_TADA)
    loss = tada_loss(raw, target, emulated_source=emulated, patch_shape=cfg.patch_shape)
    r_naive = regret_stub(pe_source_on_target=0.50, intrinsic_difficulty_target=0.01)
    r_tada = regret_stub(pe_source_on_target=0.03, intrinsic_difficulty_target=0.01)
    return {
        "emulator_kernel_sum": float(project_to_symmetric_sum_one(DENOISE_KERNEL_TADA).sum()),
        "loss": loss,
        "regret_naive_pct": r_naive * 100,
        "regret_tada_pct": r_tada * 100,
        "target_residual_energy": float(np.mean(kb_residual(target) ** 2)),
    }


def evaluation_demo(cfg: TADAConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["table_iii_rows"] = len(table_iii_target_regrets())
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_toy": table_i_toy_kernels_detailed(),
        "table_ii_flickr_targets": table_ii_flickr_targets(),
        "table_iii_regrets": table_iii_target_regrets(),
        "csm_formalization": csm_formalization(),
    }
