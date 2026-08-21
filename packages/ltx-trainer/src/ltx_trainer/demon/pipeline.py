"""DEMON framework card, demos, and benchmark bundles (arXiv:2605.28657)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.demon.config import DemonConfig
from ltx_trainer.demon.layout import LIMITATIONS
from ltx_trainer.demon.metrics import (
    table_12_propagation,
    table_13_ablation,
    table_14_depth_tradeoff,
    table_2_per_frame_curves,
    table_3_system_comparison,
)
from ltx_trainer.demon.ring_buffer import ablation_completion_rates, propagation_taxonomy, ring_smoke
from ltx_trainer.demon.sde import sde_smoke
from ltx_trainer.demon.vae_window import vae_smoke


def framework_card(cfg: DemonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DemonConfig()
    return {
        "name": "DEMON",
        "paper": cfg.paper_arxiv,
        "authors": "Ryan Fosdick (Daydream / Livepeer)",
        "idea": (
            "Real-time streaming diffusion on ACE-Step 1.5: ring-buffer denoising as a playable "
            "musical control surface with per-slot heterogeneous schedules, shared mutable per-step "
            "curves, SDE source blending, and windowed VAE decode."
        ),
        "base_model": cfg.base_model,
        "streaming": {
            "denoising_steps": cfg.denoising_steps,
            "production_depth": cfg.production_depth,
            "latent_hz": cfg.latent_hz,
            "latent_frames_60s": cfg.latent_frames_60s,
        },
        "throughput_5090": {
            "gens_per_sec_depth8": cfg.gens_per_sec_depth8,
            "gens_per_sec_depth4": cfg.gens_per_sec_depth4,
            "tick_ms_depth8": cfg.tick_ms_depth8,
            "e2e_ms_depth8_3s_window": cfg.e2e_ms_depth8_3s_window,
        },
        "contributions": [
            "Per-slot heterogeneous denoise scheduling",
            "Shared mutable per-step state (1-tick onset)",
            "Per-frame SDE source blending",
            "Windowed VAE decode (8.0× @ 3s window)",
        ],
        "propagation_classes": propagation_taxonomy(),
        "limitations": list(LIMITATIONS),
    }


def headline_results(cfg: DemonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DemonConfig()
    ab = ablation_completion_rates(cfg)
    return {
        "throughput": f"{cfg.gens_per_sec_depth8} gens/sec @ depth 8 (5090, 60s)",
        "production": f"depth {cfg.production_depth}: {cfg.gens_per_sec_depth4} gens/sec, ~{cfg.depth4_first_effect_ms:.0f} ms denoise first-effect",
        "heterogeneous_vs_reset": (
            f"Continuous slider sweep: per-slot {ab['per_slot_sweep']:.0%} vs global-reset {ab['global_reset_sweep']:.1%} completion"
        ),
        "vae_window": f"Windowed decode {cfg.vae_speedup_3s:.1f}× speedup (3s window, sample-identical interior)",
        "per_step_onset": f"SDE/shared curves onset ~{cfg.sde_curve_first_effect_ms:.0f} ms (1 tick @ depth 8)",
    }


def pipeline_demo(cfg: DemonConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or DemonConfig()
    return {
        "ring": ring_smoke(cfg),
        "sde": sde_smoke(cfg, seed=seed),
        "vae": vae_smoke(cfg),
        "gens_per_sec_depth8": cfg.gens_per_sec_depth8,
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle(cfg: DemonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DemonConfig()
    return {
        "table_2_per_frame_curves": table_2_per_frame_curves(),
        "table_3_system_comparison": table_3_system_comparison(),
        "table_12_propagation": table_12_propagation(cfg),
        "table_13_ablation": table_13_ablation(cfg),
        "table_14_depth_tradeoff": table_14_depth_tradeoff(cfg),
        "propagation_taxonomy": propagation_taxonomy(),
        "headline": headline_results(cfg),
    }
