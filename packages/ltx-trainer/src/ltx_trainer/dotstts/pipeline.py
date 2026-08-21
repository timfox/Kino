"""Framework card and benchmark tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dotstts.ar_fm import ar_fm_demo
from ltx_trainer.dotstts.audiovae import audiovae_demo
from ltx_trainer.dotstts.config import DotsttsConfig
from ltx_trainer.dotstts.meanflow import meanflow_demo
from ltx_trainer.dotstts.soar import soar_demo


def framework_card(cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "framework": cfg.framework,
        "params_b": cfg.params_b,
        "license": cfg.license,
        "github": cfg.github,
        "hf_collection": cfg.hf_collection,
        "components": [
            "semantic_causal_audiovae",
            "qwen_llm_semantic_planner",
            "full_history_ar_flow_matching",
            "soar_self_corrective_posttrain",
            "cfg_aware_meanflow_distillation",
        ],
        "streaming_modes": ["plain_tts", "1t1a_interleaved"],
        "headline": {
            "seed_avg_wer_pct": cfg.seed_avg_wer,
            "seed_avg_sim": cfg.seed_avg_sim,
            "minimax_avg_sim": cfg.minimax_avg_sim_soar,
            "ttfp_interleaved_ms": cfg.ttfp_interleaved_ms,
        },
    }


def table1_audiovae_reconstruction() -> list[dict[str, Any]]:
    """Table 1 — LibriSpeech test-other (selected rows)."""
    return [
        {
            "model": "SemanticVAE",
            "sample_rate_khz": 16,
            "fps": 40,
            "pesq_nb": 3.99,
            "stoi": 0.969,
            "sim": 0.963,
            "wer_pct": 4.15,
        },
        {
            "model": "MingTok-Audio",
            "sample_rate_khz": 16,
            "fps": 50,
            "pesq_nb": 4.23,
            "stoi": 0.981,
            "sim": 0.950,
            "wer_pct": 4.27,
        },
        {
            "model": "dots.tts VAE",
            "sample_rate_khz": 48,
            "fps": 25,
            "pesq_nb": 4.09,
            "stoi": 0.973,
            "sim": 0.969,
            "wer_pct": 4.14,
        },
    ]


def table2_seed_tts_eval() -> list[dict[str, Any]]:
    """Table 2 — Seed-TTS-Eval zero-shot (selected rows)."""
    return [
        {
            "model": "CosyVoice 3",
            "params": "1.5B",
            "en_wer": 2.22,
            "en_sim": 72.0,
            "zh_wer": 1.12,
            "zh_sim": 78.1,
            "zh_hard_wer": 5.83,
            "zh_hard_sim": 75.8,
            "avg_wer": 3.06,
            "avg_sim": 75.3,
        },
        {
            "model": "Seed-TTS",
            "params": "—",
            "en_wer": 2.25,
            "en_sim": 76.2,
            "zh_wer": 1.12,
            "zh_sim": 79.6,
            "zh_hard_wer": 7.59,
            "zh_hard_sim": 77.6,
            "avg_wer": 3.65,
            "avg_sim": 77.8,
        },
        {
            "model": "VoxCPM 2",
            "params": "2B",
            "en_wer": 1.84,
            "en_sim": 75.3,
            "zh_wer": 0.97,
            "zh_sim": 79.5,
            "zh_hard_wer": 8.13,
            "zh_hard_sim": 75.3,
            "avg_wer": 3.65,
            "avg_sim": 76.7,
        },
        {
            "model": "dots.tts (Pretrain)",
            "params": "2B",
            "en_wer": 1.34,
            "en_sim": 76.8,
            "zh_wer": 0.96,
            "zh_sim": 80.5,
            "zh_hard_wer": 6.46,
            "zh_hard_sim": 79.2,
            "avg_wer": 2.92,
            "avg_sim": 78.8,
        },
        {
            "model": "dots.tts (SOAR)",
            "params": "2B",
            "en_wer": 1.30,
            "en_sim": 77.1,
            "zh_wer": 0.94,
            "zh_sim": 81.0,
            "zh_hard_wer": 6.60,
            "zh_hard_sim": 79.5,
            "avg_wer": 2.95,
            "avg_sim": 79.2,
        },
        {
            "model": "dots.tts (MF, NFE=4)",
            "params": "2B",
            "en_wer": 1.29,
            "en_sim": 76.2,
            "zh_wer": 0.94,
            "zh_sim": 80.0,
            "zh_hard_wer": 6.60,
            "zh_hard_sim": 78.5,
            "avg_wer": 2.94,
            "avg_sim": 78.2,
        },
    ]


def table3_minimax_multilingual_average() -> list[dict[str, Any]]:
    """Table 3 — 24-language average WER/SIM."""
    return [
        {"model": "MiniMax-Speech", "avg_wer": 2.8, "avg_sim": 76.6},
        {"model": "Fish-Audio S2", "avg_wer": 3.7, "avg_sim": 78.0},
        {"model": "VoxCPM 2", "avg_wer": 5.7, "avg_sim": 82.3},
        {"model": "dots.tts (Pretrain)", "avg_wer": 6.6, "avg_sim": 83.5},
        {"model": "dots.tts (SOAR)", "avg_wer": 6.8, "avg_sim": 83.9},
        {"model": "dots.tts (MF4)", "avg_wer": 6.8, "avg_sim": 83.5},
    ]


def table5_emergent_tts_selected() -> list[dict[str, Any]]:
    """Table 5 — EmergentTTS-Eval selected open-source rows."""
    return [
        {
            "model": "gpt-4o-mini-tts (baseline)",
            "wer": 10.61,
            "overall": 50.0,
            "syntax": 57.1,
        },
        {
            "model": "dots.tts (Pretrain)",
            "wer": 10.86,
            "overall": 49.2,
            "syntax": 58.4,
        },
        {
            "model": "dots.tts (SOAR)",
            "wer": 10.45,
            "overall": 47.6,
            "syntax": 65.7,
        },
        {
            "model": "Qwen3-TTS",
            "wer": 17.32,
            "overall": 42.8,
            "syntax": 60.4,
        },
        {
            "model": "VoxCPM2",
            "wer": 11.84,
            "overall": 41.1,
            "syntax": 52.3,
        },
    ]


def table_efficiency() -> list[dict[str, Any]]:
    """§3.4 — inference efficiency (MF NFE=4, H800)."""
    return [
        {
            "mode": "plain",
            "ttfp_ms": 85.4,
            "rtf": 0.231,
            "interleaved": False,
        },
        {
            "mode": "1T1A interleaved",
            "ttfp_ms": 54.4,
            "rtf": 0.245,
            "interleaved": True,
        },
    ]


def training_stages() -> list[dict[str, Any]]:
    """§3.2 — training pipeline summary."""
    return [
        {"stage": "AudioVAE S1", "steps_k": 500, "focus": "reconstruction"},
        {"stage": "AudioVAE S2", "steps_k": 200, "focus": "WavLM + multitask"},
        {"stage": "Backbone align", "steps_k": 100, "focus": "frozen LLM, Emilia"},
        {"stage": "Backbone general", "steps_k": 700, "focus": "1.5M h full mix"},
        {"stage": "Backbone anneal", "steps_k": 100, "focus": "quality subset"},
        {"stage": "SOAR", "steps_k": 50, "focus": "AR-FM DiT only"},
        {"stage": "MeanFlow", "steps_k": 50, "focus": "2–4 NFE student"},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_audiovae_reconstruction": table1_audiovae_reconstruction(),
        "table2_seed_tts_eval": table2_seed_tts_eval(),
        "table3_minimax_multilingual_average": table3_minimax_multilingual_average(),
        "table5_emergent_tts_selected": table5_emergent_tts_selected(),
        "table_efficiency": table_efficiency(),
        "training_stages": training_stages(),
    }


def headline_results(cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    return {
        "seed_avg_wer_pct": cfg.seed_avg_wer,
        "seed_avg_sim": cfg.seed_avg_sim,
        "seed_zh_sim": cfg.seed_zh_sim,
        "minimax_avg_sim": cfg.minimax_avg_sim_soar,
        "emergent_syntax_pct": cfg.emergent_syntax,
        "ttfp_interleaved_ms": cfg.ttfp_interleaved_ms,
        "vae_wer_pct": cfg.vae_wer_pct,
    }


def evaluation_demo(*, seed: int = 0, cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    return {
        "framework": framework_card(cfg),
        "audiovae": audiovae_demo(seed=seed, cfg=cfg),
        "ar_fm": ar_fm_demo(seed=seed, cfg=cfg),
        "soar": soar_demo(seed=seed, cfg=cfg),
        "meanflow": meanflow_demo(seed=seed, cfg=cfg),
        "headline": headline_results(cfg),
    }
