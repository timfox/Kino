"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.glass.config import GlassConfig
from ltx_trainer.glass.grpo import combined_reward, group_advantages, min_max_normalize
from ltx_trainer.glass.lora_arith import compose_axes, interpolate_opposite


def framework_card(cfg: GlassConfig | None = None) -> dict[str, Any]:
    c = cfg or GlassConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "composable_acoustic_style_control",
        "backbone": c.backbone,
        "training": "GRPO on frozen AR-TTS + one LoRA per style direction",
        "inference": ["LoRA swapping", "interpolation", "multi-axis composition"],
        "headline": headline_results(c),
    }


def headline_results(cfg: GlassConfig | None = None) -> dict[str, Any]:
    c = cfg or GlassConfig()
    return {
        "baseline_wer": c.baseline_wer,
        "fast_sps": c.fast_sps,
        "slow_sps": c.slow_sps,
        "interp_wer_min_speed": c.interp_speed_wer_min,
        "interp_wer_min_pitch": c.interp_pitch_wer_min,
        "lora_params_m": c.lora_params_m,
        "lora_fraction_pct": c.lora_fraction_pct,
        "compose_weight_default": c.compose_weight_default,
    }


def table1_individual_control(cfg: GlassConfig | None = None) -> list[dict[str, Any]]:
    """Table 1 — individual style control on Seed-TTS-eval test_en."""
    c = cfg or GlassConfig()
    return [
        {
            "method": "Baseline (CosyVoice2-0.5B)",
            "sps": c.baseline_sps,
            "f0_m": c.baseline_f0_m,
            "f0_f": c.baseline_f0_f,
            "wer": c.baseline_wer,
            "spksim": c.baseline_spksim,
            "utmos": c.baseline_utmos,
        },
        {
            "method": "Fast LoRA (ours)",
            "sps": c.fast_sps,
            "f0_m": 120.4,
            "f0_f": 191.0,
            "wer": c.fast_wer,
            "spksim": c.fast_spksim,
            "utmos": 3.30,
        },
        {
            "method": "Slow LoRA (ours)",
            "sps": c.slow_sps,
            "f0_m": 122.1,
            "f0_f": 194.3,
            "wer": c.slow_wer,
            "spksim": c.slow_spksim,
            "utmos": 3.05,
        },
        {
            "method": "High-pitch LoRA (ours)",
            "sps": 3.61,
            "f0_m": c.high_f0_m,
            "f0_f": c.high_f0_f,
            "wer": c.high_wer,
            "spksim": 0.609,
            "utmos": 3.37,
        },
        {
            "method": "Low-pitch LoRA (ours)",
            "sps": 3.68,
            "f0_m": c.low_f0_m,
            "f0_f": c.low_f0_f,
            "wer": c.low_wer,
            "spksim": 0.632,
            "utmos": 3.16,
        },
    ]


def table2_interpolation(cfg: GlassConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — continuous interpolation sweep (200-utterance subset)."""
    c = cfg or GlassConfig()
    return [
        {"axis": "speed", "alpha": 0.50, "sps": 4.00, "wer": c.interp_speed_wer_min},
        {"axis": "speed", "alpha": 0.00, "sps": 2.30, "wer": 4.50},
        {"axis": "speed", "alpha": 1.00, "sps": 5.52, "wer": 3.51},
        {"axis": "pitch", "alpha": 0.50, "sps": 3.64, "wer": c.interp_pitch_wer_min},
        {"axis": "pitch", "alpha": 0.00, "sps": 3.72, "wer": 3.36},
        {"axis": "pitch", "alpha": 1.00, "sps": 3.62, "wer": 2.96},
    ]


def benchmarks_bundle(cfg: GlassConfig | None = None) -> dict[str, Any]:
    c = cfg or GlassConfig()
    return {
        "table1_individual": table1_individual_control(c),
        "table2_interpolation": table2_interpolation(c),
        "style_axes": ["speed_fast", "speed_slow", "pitch_high", "pitch_low"],
        "eval_set": "Seed-TTS-eval test_en (N=1088)",
    }


def pipeline_demo(seed: int = 42, cfg: GlassConfig | None = None) -> dict[str, Any]:
    """CPU stub: GRPO rewards + LoRA interpolation/composition."""
    c = cfg or GlassConfig()
    rng = np.random.default_rng(seed)

    token_lengths = rng.integers(80, 140, size=c.grpo_group_size)
    style_norm = min_max_normalize(token_lengths.astype(float))
    fast_rewards = [
        combined_reward(wer=0.03, style_score=1.0 - float(s), eta=c.reward_eta, gamma=c.wer_gamma)
        for s in style_norm
    ]
    advantages = group_advantages(np.array(fast_rewards))

    fast_delta = rng.standard_normal((4, 4)) * 0.01
    slow_delta = -fast_delta * 0.9
    high_delta = rng.standard_normal((4, 4)) * 0.008
    low_delta = -high_delta * 0.85

    alpha = 0.5
    speed_blend = interpolate_opposite(fast_delta, slow_delta, alpha)
    composed = compose_axes(speed_blend, high_delta, w_speed=0.5, w_pitch=0.5)

    return {
        "seed": seed,
        "grpo_group_size": c.grpo_group_size,
        "advantages_mean": float(np.mean(advantages)),
        "fast_reward_mean": float(np.mean(fast_rewards)),
        "lora_fraction_pct": c.lora_fraction_pct,
        "interpolated_shape": list(speed_blend.shape),
        "composed_shape": list(composed.shape),
        "paper_fast_sps": c.fast_sps,
        "paper_slow_sps": c.slow_sps,
        "interp_wer_min": min(c.interp_speed_wer_min, c.interp_pitch_wer_min),
    }


def evaluation_demo(seed: int = 42, cfg: GlassConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(seed=seed, cfg=cfg)
    c = cfg or GlassConfig()
    return {**demo, "backbone": c.backbone}
