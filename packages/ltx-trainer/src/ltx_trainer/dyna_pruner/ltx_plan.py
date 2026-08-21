"""GOPEX / LTX integration hooks for input-adaptive co-pruning."""

from __future__ import annotations

import os
from typing import Any

from ltx_trainer.dyna_pruner.config import DynaPrunerConfig


def ltx_prep_plan(
    *,
    project: str | None = None,
    sd: float | None = None,
    sw: float | None = None,
) -> dict[str, Any]:
    """
    Recommend spatial masking before VAE encode when clips have large static regions.

    Env: ``GOPEX_DYNA_PRUNER_SD``, ``GOPEX_DYNA_PRUNER_SW``, ``GOPEX_PREP_PROJECTS``.
    """
    cfg = DynaPrunerConfig()
    proj = project or os.environ.get("GOPEX_PREP_PROJECTS", "merged_native").split(",")[0]
    data_sd = sd if sd is not None else float(os.environ.get("GOPEX_DYNA_PRUNER_SD", str(cfg.data_sparsity_sd)))
    model_sw = sw if sw is not None else float(os.environ.get("GOPEX_DYNA_PRUNER_SW", str(cfg.model_sparsity_sw)))
    return {
        "paper": cfg.paper_arxiv,
        "project": proj,
        "hook": "pre_vae_spatial_mask",
        "description": (
            "Run mask generator on each clip; skip or zero calm ERP/ocean tiles before "
            "gopex_prep_parallel latents-once to cut encode FLOPs on redundant frames."
        ),
        "env": {
            "GOPEX_DYNA_PRUNER": "1",
            "GOPEX_DYNA_PRUNER_SD": str(data_sd),
            "GOPEX_DYNA_PRUNER_SW": str(model_sw),
        },
        "targets": [
            "gopex_prep_parallel.py latents-once",
            "kino-parallel-prep.sh latents-once",
            "sphere360 / weather_live spatio-temporal tiles",
        ],
        "expected_encode_reduction_pct": round(cfg.typical_gflops_reduction_pct * 0.6, 1),
    }


def ltx_inference_plan(
    *,
    backbone: str = "LTX-2",
    sd: float | None = None,
    sw: float | None = None,
) -> dict[str, Any]:
    """Per-sample sparse DiT sub-network at delivery/inference."""
    cfg = DynaPrunerConfig()
    data_sd = sd if sd is not None else cfg.data_sparsity_sd
    model_sw = sw if sw is not None else cfg.model_sparsity_sw
    return {
        "backbone": backbone,
        "mode": "input_adaptive_subnetwork",
        "steps": [
            "Forward mask generator on conditioning frames → S",
            "Binarize M_data / M_weight (STE-trained weights)",
            "Apply structured filter mask on conv/attn blocks",
            "Run denoiser on masked latents only",
        ],
        "sd": data_sd,
        "sw": model_sw,
        "expected_latency_reduction_pct": cfg.typical_latency_reduction_pct,
        "compatible_backbones": list(cfg.backbones),
        "edge_target": "NVIDIA Jetson AGX Orin (paper Table I latency)",
    }


def energy_aware_training_note(cfg: DynaPrunerConfig | None = None) -> dict[str, Any]:
    """Attach to energy_aware LTX OpEx plans as a green-AI compression lever."""
    cfg = cfg or DynaPrunerConfig()
    return {
        "strategy": "dyna_pruner_co_pruning",
        "effective_flops_factor": round(1.0 - cfg.typical_gflops_reduction_pct / 100.0, 3),
        "effective_power_factor": round(1.0 - cfg.typical_latency_reduction_pct / 100.0, 3),
        "accuracy_tradeoff_pct_mse": cfg.typical_mse_increase_pct,
        "use_when": [
            "Spatio-temporal prediction or video prep with static backgrounds",
            "Edge delivery after LTX train (Jetson / IoT)",
            "Carbon-aware scheduling with elasticity headroom",
        ],
    }


def gopex_env_exports() -> dict[str, str]:
    cfg = DynaPrunerConfig()
    return {
        "GOPEX_DYNA_PRUNER": "1",
        "GOPEX_DYNA_PRUNER_SD": str(cfg.data_sparsity_sd),
        "GOPEX_DYNA_PRUNER_SW": str(cfg.model_sparsity_sw),
    }
