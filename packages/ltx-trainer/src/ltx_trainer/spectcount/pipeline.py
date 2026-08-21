"""Framework card and benchmark tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spectcount.config import SpectCountConfig
from ltx_trainer.spectcount.probing import probing_demo
from ltx_trainer.spectcount.signals import signal_demo


def framework_card(cfg: SpectCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpectCountConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "framework": cfg.framework,
        "backbones": [cfg.backbone_af3, cfg.backbone_qwen2],
        "components": [
            "probing_detectability_analysis",
            "on_the_fly_synthetic_pulse_counting",
            "lora_sft_all_modules",
        ],
        "signal_generation": {
            "n_max": cfg.n_max,
            "cmel": cfg.cmel,
            "t_total_s": cfg.t_total_s,
            "fully_synthetic": True,
        },
        "headline": {
            "mmau_mini_total": cfg.mmau_mini_total,
            "mmau_mini_gain": round(cfg.mmau_mini_total - cfg.mmau_mini_base, 2),
            "mmar_gain_pct": round(100 * (cfg.mmar - cfg.mmar_base) / cfg.mmar_base, 2),
        },
    }


def table1_benchmarks() -> list[dict[str, Any]]:
    """Table 1 — auditory understanding benchmarks (%)."""
    return [
        {
            "model": "Audio Flamingo 3",
            "setting": "Base (reproduced)",
            "mmau_mini_sound": 81.08,
            "mmau_mini_music": 71.86,
            "mmau_mini_speech": 68.77,
            "mmau_mini_total": 73.90,
            "mmau_test_total": 72.36,
            "mmar": 52.90,
            "mmsu": 61.92,
            "air_bench": 64.16,
        },
        {
            "model": "Audio Flamingo 3",
            "setting": "SpectCount",
            "mmau_mini_sound": 83.18,
            "mmau_mini_music": 77.54,
            "mmau_mini_speech": 74.47,
            "mmau_mini_total": 78.40,
            "mmau_test_total": 73.79,
            "mmar": 56.30,
            "mmsu": 63.18,
            "air_bench": 64.85,
        },
        {
            "model": "Qwen2-Audio-Instruct",
            "setting": "Base (reproduced)",
            "mmau_mini_sound": 66.67,
            "mmau_mini_music": 57.19,
            "mmau_mini_speech": 50.75,
            "mmau_mini_total": 58.20,
            "mmau_test_total": 56.44,
            "mmar": 40.10,
            "mmsu": 48.44,
            "air_bench": 60.17,
        },
        {
            "model": "Qwen2-Audio-Instruct",
            "setting": "SpectCount",
            "mmau_mini_sound": 70.57,
            "mmau_mini_music": 58.38,
            "mmau_mini_speech": 61.86,
            "mmau_mini_total": 63.60,
            "mmau_test_total": 61.29,
            "mmar": 45.70,
            "mmsu": 54.24,
            "air_bench": 62.78,
        },
    ]


def table3_ablation() -> list[dict[str, Any]]:
    """Table 3 — task formulation and module ablation (% MMAU-test-mini)."""
    return [
        {"setting": "Freq-axis discrimination only", "accuracy": 74.7},
        {"setting": "Time-axis aggregation only", "accuracy": 77.2},
        {"setting": "Audio encoder LoRA only", "accuracy": 75.2},
        {"setting": "LLM backbone LoRA only", "accuracy": 77.0},
        {"setting": "SpectCount (full)", "accuracy": 78.4},
    ]


def table2_signal_config(cfg: SpectCountConfig | None = None) -> dict[str, Any]:
    """Table 2 — signal generation configuration."""
    cfg = cfg or SpectCountConfig()
    return {
        "sample_rate_hz": cfg.sample_rate_hz,
        "n_max": cfg.n_max,
        "cmel": cfg.cmel,
        "t_min_ms": cfg.t_min_ms,
        "t_max_ms": cfg.t_max_ms,
        "attack_ms": cfg.attack_ms,
        "release_ms": cfg.release_ms,
        "t_gap_ms": cfg.t_gap_ms,
        "t_total_s": cfg.t_total_s,
        "amp_range": [cfg.amp_min, cfg.amp_max],
        "noise_range": [cfg.noise_min, cfg.noise_max],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_auditory_benchmarks": table1_benchmarks(),
        "table2_signal_config": table2_signal_config(),
        "table3_ablation": table3_ablation(),
    }


def headline_results(cfg: SpectCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpectCountConfig()
    rel_mini = round(100 * (cfg.mmau_mini_total - cfg.mmau_mini_base) / cfg.mmau_mini_base, 2)
    rel_test = round(100 * (cfg.mmau_test_total - cfg.mmau_test_base) / cfg.mmau_test_base, 2)
    return {
        "mmau_mini_total": cfg.mmau_mini_total,
        "relative_gain_mmau_mini_pct": rel_mini,
        "relative_gain_mmau_test_pct": rel_test,
        "mmar": cfg.mmar,
        "best_ablation": cfg.mmau_mini_total,
    }


def evaluation_demo(*, seed: int = 0, cfg: SpectCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpectCountConfig()
    return {
        "framework": framework_card(cfg),
        "signal": signal_demo(seed=seed, cfg=cfg),
        "probing": probing_demo(seed=seed, cfg=cfg),
        "headline": headline_results(cfg),
    }
