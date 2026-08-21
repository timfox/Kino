"""Paper table anchors (DEMON arXiv:2605.28657)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.demon.config import DemonConfig
from ltx_trainer.demon.sde import PER_FRAME_CURVES


def table_3_system_comparison() -> list[dict[str, Any]]:
    return [
        {
            "system": "DEMON (ours)",
            "architecture": "Diffusion DiT + TRT",
            "tick_ms": 81,
            "per_frame_resolution_ms": 40,
            "interactive_control": "Per-frame denoising curves + multi-condition",
            "hardware": "RTX 5090",
        },
        {
            "system": "Lyria RealTime",
            "architecture": "Autoregressive",
            "tick_ms": 2000,
            "per_frame_resolution_ms": None,
            "interactive_control": "Tempo, brightness, density, stems, key (chunk boundary)",
            "hardware": "Cloud API",
        },
    ]


def table_12_propagation(cfg: DemonConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or DemonConfig()
    return [
        {"parameter": "Denoise (1.0→0.5)", "first_effect_ms": cfg.denoise_first_effect_ms, "class": "per_request"},
        {"parameter": "Prompt switch", "first_effect_ms": cfg.denoise_first_effect_ms, "class": "per_request"},
        {"parameter": "SDE curve (shared)", "first_effect_ms": cfg.sde_curve_first_effect_ms, "class": "per_step_shared"},
        {"parameter": "x0 target strength (shared)", "first_effect_ms": cfg.sde_curve_first_effect_ms, "class": "per_step_shared"},
        {"parameter": "LoRA refit", "first_effect_ms": cfg.tick_ms_depth8, "class": "model_weight"},
    ]


def table_13_ablation(cfg: DemonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DemonConfig()
    return {
        "per_slot_completion_sweep": cfg.per_slot_completion_rate_sweep,
        "global_reset_completion_sweep": cfg.global_reset_completion_rate_sweep,
        "per_slot_dead_air_ticks": 0,
        "global_reset_dead_air_ticks": cfg.max_depth,
        "first_new_effect_tick": cfg.max_depth,
    }


def table_14_depth_tradeoff(cfg: DemonConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or DemonConfig()
    return [
        {"depth": 1, "tick_ms": 14.0, "gens_per_sec": 8.9, "denoise_first_effect_ms": 112.0},
        {"depth": 2, "tick_ms": 24.3, "gens_per_sec": 10.3, "denoise_first_effect_ms": 219.0},
        {"depth": cfg.production_depth, "tick_ms": cfg.tick_ms_depth4, "gens_per_sec": cfg.gens_per_sec_depth4, "denoise_first_effect_ms": cfg.depth4_first_effect_ms},
        {"depth": cfg.max_depth, "tick_ms": cfg.tick_ms_depth8, "gens_per_sec": cfg.gens_per_sec_depth8, "denoise_first_effect_ms": cfg.depth8_first_effect_ms},
    ]


def table_2_per_frame_curves() -> list[dict[str, str]]:
    return [{"curve": c, "resolution": "per-frame @ 25Hz", "mutable": "shared per-step"} for c in PER_FRAME_CURVES]
