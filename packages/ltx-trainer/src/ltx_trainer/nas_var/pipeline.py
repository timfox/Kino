"""Framework card and benchmarks for NAS-VAR MRI reconstruction."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nas_var.config import NasVarConfig
from ltx_trainer.nas_var.layout import ARCHITECTURE_NOTES, BASELINES, LIMITATIONS, SCALE_CHAIN
from ltx_trainer.nas_var.mock import evaluation_smoke, scale_factorization_demo
from ltx_trainer.nas_var.tables import (
    supp_table2_mean_perceptual,
    table1_cartesian_x,
    table2_radial,
    table3_cartesian_y,
    table4_ablations_summary,
    table5_distillation_cartesian_x,
)


def framework_card(cfg: NasVarConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NasVarConfig()
    return {
        "name": "NAS-VAR",
        "paper": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "institution": cfg.institution,
        "summary": (
            "Discrete autoregressive MRI reconstruction via next-acceleration-scale prediction "
            "in a multi-input AQ-VAE token hierarchy, with on-policy privileged distillation."
        ),
        "problem": (
            "Extreme undersampling (R=32) is ill-posed in pixel space; discrete VAR tokens "
            "constrain plausible anatomy and preserve high-frequency structure."
        ),
        "components": {
            "aq_vae": {
                "codebook_size": cfg.codebook_size,
                "latent_dim": cfg.latent_dim,
                "token_grids": list(cfg.token_grids_per_scale),
                "loss_weights": dict(cfg.aq_vae_loss_weights or {}),
            },
            "transformer": {
                "depth": cfg.transformer_depth,
                "embed_dim": cfg.transformer_embed_dim,
                "heads": cfg.transformer_heads,
                "cross_attention_encoder_resolutions": [64, 32, 16],
            },
            "distillation": {
                "type": "on-policy privileged information",
                "privileged_context": "fully sampled MR image (training only)",
                "objective": "reverse KL student || teacher",
            },
        },
        "scale_chain": list(SCALE_CHAIN),
        "acceleration_scales": list(cfg.acceleration_scales),
        "inference_input_R": cfg.inference_scale,
        "evaluation": {
            "benchmark": cfg.benchmark,
            "R": cfg.eval_acceleration,
            "contrasts": list(cfg.contrasts),
            "masks": list(cfg.sampling_masks),
            "metrics": ["PSNR", "SSIM", "LPIPS", "VGG-LPIPS", "DISTS"],
        },
        "findings": [
            "Best LPIPS across masks/contrasts at R=32; strong on Cartesian (harder artifacts).",
            "FLAIR PSNR +2.76 dB vs MambaRecon on Cartesian-X (18.53 -> 21.29).",
            "Privileged distillation improves PSNR/SSIM; suppresses hallucinated anatomy.",
            f"Inference ~{cfg.inference_seconds_per_image_a5000}s/image on RTX A5000 (argmax).",
        ],
        "baselines": list(BASELINES),
        "architecture_notes": list(ARCHITECTURE_NOTES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_cartesian_x": table1_cartesian_x(),
        "table2_radial": table2_radial(),
        "table3_cartesian_y": table3_cartesian_y(),
        "table4_ablations": table4_ablations_summary(),
        "table5_distillation": table5_distillation_cartesian_x(),
        "supp_table2_perceptual": supp_table2_mean_perceptual(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "scale_chain": scale_factorization_demo()}
