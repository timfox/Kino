"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deblur_nvs.benchmarks import benchmarks_bundle, table2_ours
from ltx_trainer.deblur_nvs.config import (
    DeblurNVSConfig,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
)
from ltx_trainer.deblur_nvs.dataset import dataset_card
from ltx_trainer.deblur_nvs.integration import gopex_links, ltx_plan_stub, ltx_prep_notes


def framework_card(cfg: DeblurNVSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeblurNVSConfig()
    ours = table2_ours()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "project_url": PROJECT_URL,
        "problem": (
            "Motion blur corrupts local detail and cross-view correspondences; per-scene Deblur-NeRF/BAGS "
            "is slow, and cascaded 2D deblur + GLD is multi-view inconsistent."
        ),
        "method": {
            "context_stage": "LoRA DA3 encoder + camera-free latent diffusion restores sharp context latents (Eq. 7–9)",
            "target_stage": "Camera-conditioned latent diffusion + RGB decoder for novel view (Eq. 11–14)",
            "dataset": "DL3DV-10K-Blur: 8× interp + temporal average, N∈{5,7,9,11}",
            "inference": f"Context {cfg.context_steps} + target {cfg.target_steps} Euler steps @ {cfg.height}×{cfg.width}",
            "backbone": "GLD single-level DA3 geometric latent space",
        },
        "results_table2": {
            "lpips": ours["lpips"],
            "fid": ours["fid"],
            "psnr": ours["psnr"],
            "time": ours["time"],
            "vs_gld_lpips": ours["lpips"] < 0.503,
        },
        "dataset_card": dataset_card(),
        "reference_metrics": benchmarks_bundle(),
        "gopex_links": gopex_links(),
        "ltx_plan": ltx_plan_stub(),
        "integration": (
            "GOPEX implements blur synthesis, two-stage latent stubs, and Tables I–VI anchors. "
            "Wire PKU DeblurNVS weights for production; complements gphotos sparse prep and "
            "ERP NVS stacks (ErpGS, Gimbal360)."
        ),
        "ltx_notes": ltx_prep_notes(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    from ltx_trainer.deblur_nvs.pipeline import evaluation_demo_run

    cfg = DeblurNVSConfig(height=48, width=80, context_views=3)
    return evaluation_demo_run(cfg, device="cpu", seed=seed)
