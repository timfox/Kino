"""Framework card and paper benchmark excerpts (arXiv:2605.23463)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stepaudio25.config import Stepaudio25Config
from ltx_trainer.stepaudio25.layout import LIMITATIONS
from ltx_trainer.stepaudio25.mock import evaluation_smoke


def framework_card(cfg: Stepaudio25Config | None = None) -> dict[str, Any]:
    cfg = cfg or Stepaudio25Config()
    return {
        "name": "StepAudio 2.5 — unified audio-language foundation (ASR + TTS + Realtime)",
        "paper": cfg.paper_arxiv,
        "family": cfg.family,
        "thesis": (
            "Once text and audio share a multimodal space, task differences reduce to "
            "data construction, optimization targets, and decoding constraints."
        ),
        "backbone": cfg.backbone,
        "branches": {
            "asr": "MTP-5 verifiable multi-token decoding on shared decoder",
            "tts": "Audio tokens as a language; SFT + preference RLHF with generative reward model",
            "realtime": "Progressive SFT (persona/paralinguistics) + PPO-style RLHF with rubrics",
        },
        "pretraining": cfg.pretrain_tokens,
        "headlines": {
            "asr_zh_avg_cer_pct": cfg.asr_zh_avg_cer,
            "asr_en_avg_wer_pct": cfg.asr_en_avg_wer,
            "asr_long_avg_er_pct": cfg.asr_long_avg_er,
            "asr_rtf": cfg.asr_rtf,
            "tts_arena_win_rate": cfg.tts_arena_win_rate,
            "realtime_human_eval_margin": cfg.realtime_human_margin,
            "realtime_spqa_margin": cfg.realtime_spqa_margin,
        },
        "limitations": LIMITATIONS,
    }


def table_1_asr_error_rates() -> list[dict[str, Any]]:
    """Table 1 excerpt: StepAudio 2.5 ASR vs baselines (error rate %)."""
    return [
        {
            "test_set": "AISHELL-1",
            "category": "Chinese",
            "qwen3_asr_1_7b": 1.49,
            "stepaudio25_asr": 0.71,
            "stepaudio25_asr_wo_mtp": 0.79,
        },
        {
            "test_set": "WenetSpeech testnet",
            "category": "Chinese",
            "qwen3_asr_1_7b": 4.44,
            "stepaudio25_asr": 4.54,
            "stepaudio25_asr_wo_mtp": 4.57,
        },
        {
            "test_set": "Chinese average",
            "category": "Chinese",
            "qwen3_asr_1_7b": 3.17,
            "stepaudio25_asr": 2.97,
            "stepaudio25_asr_wo_mtp": 3.00,
        },
        {
            "test_set": "LibriSpeech clean",
            "category": "English",
            "qwen3_asr_1_7b": 1.69,
            "stepaudio25_asr": 1.38,
            "stepaudio25_asr_wo_mtp": 1.40,
        },
        {
            "test_set": "English average",
            "category": "English",
            "qwen3_asr_1_7b": 3.85,
            "stepaudio25_asr": 3.68,
            "stepaudio25_asr_wo_mtp": 3.83,
        },
        {
            "test_set": "LibriSpeech clean long",
            "category": "Long-form",
            "qwen3_asr_1_7b": 1.95,
            "stepaudio25_asr": 1.27,
            "stepaudio25_asr_wo_mtp": 1.27,
        },
        {
            "test_set": "Long-form average",
            "category": "Long-form",
            "qwen3_asr_1_7b": 4.20,
            "stepaudio25_asr": 3.70,
            "stepaudio25_asr_wo_mtp": 3.63,
        },
    ]


def table_2_asr_rtf() -> list[dict[str, Any]]:
    """Table 2: real-time factor on 30s clips (single H800, single concurrency)."""
    return [
        {"model": "VibeVoice-ASR", "rtf": 0.1039},
        {"model": "FunASR-Nano", "rtf": 0.0591},
        {"model": "Doubao-ASR-2603", "rtf": 0.0640},
        {"model": "Qwen3-ASR-1.7B", "rtf": 0.0094},
        {"model": "StepAudio 2.5 ASR", "rtf": 0.0053},
    ]


def table_3_mtp_acceptance() -> list[dict[str, Any]]:
    """Table 3: strict per-position MTP acceptance (WenetSpeech meeting)."""
    return [
        {
            "config": "MTP-3",
            "positions": [0.96, 0.88, 0.80],
            "avg_accepted_length": "3.6 / 4",
        },
        {
            "config": "MTP-5",
            "positions": [0.95, 0.88, 0.80, 0.71, 0.64],
            "avg_accepted_length": "5.0 / 6",
        },
        {
            "config": "MTP-7",
            "positions": [0.96, 0.88, 0.80, 0.72, 0.65, 0.59, 0.53],
            "avg_accepted_length": "6.1 / 8",
        },
    ]


def figure_4_tts_arena() -> dict[str, Any]:
    """Figure 4 headline: pairwise arena win rates."""
    return {
        "overall_win_rate": 0.676,
        "baselines": ["MiniMax-2.8-HD", "Elevenlabs-v3", "Gemini-3.1-Flash-TTS"],
        "prompts": 774,
        "note": "Higher win rate vs each baseline in arena-style human preference judging.",
    }


def figure_5_realtime_eval() -> dict[str, Any]:
    """Figure 5 headline margins (subjective + objective suites)."""
    return {
        "suites": [
            "Step-Dialogue-Human-Eval",
            "step_Dialogue_general",
            "step-Dialogue-car",
            "Step-Dialogue-Understanding",
            "Step-SPQA",
        ],
        "margins_vs_next_best": {
            "subjective_human_eval": 10.0,
            "step_spqa": 16.6,
        },
        "note": "Realtime branch preserves reasoning while improving persona and paralinguistic dialogue.",
    }


def headline_results() -> dict[str, Any]:
    cfg = Stepaudio25Config()
    return {
        "asr_zh_avg_cer": cfg.asr_zh_avg_cer,
        "asr_en_avg_wer": cfg.asr_en_avg_wer,
        "asr_long_avg_er": cfg.asr_long_avg_er,
        "asr_rtf": cfg.asr_rtf,
        "tts_arena_win_rate": cfg.tts_arena_win_rate,
        "realtime_human_margin": cfg.realtime_human_margin,
        "realtime_spqa_margin": cfg.realtime_spqa_margin,
    }


def evaluation_demo(cfg: Stepaudio25Config | None = None) -> dict[str, Any]:
    cfg = cfg or Stepaudio25Config()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_1_asr_error_rates": table_1_asr_error_rates(),
        "table_2_asr_rtf": table_2_asr_rtf(),
        "table_3_mtp_acceptance": table_3_mtp_acceptance(),
        "figure_4_tts_arena": figure_4_tts_arena(),
        "figure_5_realtime_eval": figure_5_realtime_eval(),
        "headlines": headline_results(),
    }
