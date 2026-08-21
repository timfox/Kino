"""F5-TTS-DPS framework card and WildSpoof tables (arXiv:2605.23859)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.f5_tts_dps.config import F5TtsDpsConfig
from ltx_trainer.f5_tts_dps.dps import AUDIO_SCORING_PROMPT_HEAD, TEXT_SCORING_PROMPT_HEAD
from ltx_trainer.f5_tts_dps.layout import LIMITATIONS
from ltx_trainer.f5_tts_dps.mock import evaluation_smoke


def framework_card(cfg: F5TtsDpsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or F5TtsDpsConfig()
    return {
        "name": "F5-TTS-DPS",
        "paper": cfg.paper_arxiv,
        "challenge": cfg.challenge,
        "authors": "Renhe Sun, Jiayi Zhou, Haolin He, Yueying Feng, Jian Liu (Ant Group)",
        "base_model": cfg.base_model,
        "innovations": [
            "EMA-stabilized full-parameter SFT on TITW in-the-wild data",
            "Dual-scoring prompt selection (LALM audio + LLM semantic alignment)",
        ],
        "scorers": {
            "audio": cfg.lalm_audio_scorer,
            "text": cfg.llm_text_scorer,
        },
        "training": {
            "ema_beta": cfg.ema_beta,
            "epochs": cfg.train_epochs,
            "learning_rate": cfg.learning_rate,
            "data": "TITW-easy + TITW-hard",
        },
        "headline_metrics": {
            "dev_utmos": cfg.dev_utmos,
            "dev_wer_pct": cfg.dev_wer,
            "dev_spk_sim": cfg.dev_spk_sim,
            "test_adcf": [cfg.test_adcf_t01, cfg.test_adcf_t02, cfg.test_adcf_t08],
        },
        "prompts": {
            "audio_scoring": AUDIO_SCORING_PROMPT_HEAD,
            "text_scoring": TEXT_SCORING_PROMPT_HEAD,
        },
        "limitations": LIMITATIONS,
    }


def table_i_dev_ablation() -> list[dict[str, Any]]:
    """Table 1 — development set ablation (WildSpoof 2026 TTS track)."""
    return [
        {
            "configuration": "CosyVoice2",
            "utmos": 3.65,
            "dnsmos": 2.79,
            "wer": 7.76,
            "spk_sim": 0.403,
            "sds": 0.343,
        },
        {
            "configuration": "baseline (F5-TTS)",
            "utmos": 3.06,
            "dnsmos": 2.91,
            "wer": 12.31,
            "spk_sim": 0.450,
            "sds": 0.283,
        },
        {
            "configuration": "+ SFT",
            "utmos": 3.06,
            "dnsmos": 2.54,
            "wer": 10.60,
            "spk_sim": 0.489,
            "sds": 0.226,
        },
        {
            "configuration": "+ SFT + EMA",
            "utmos": 3.18,
            "dnsmos": 2.61,
            "wer": 9.32,
            "spk_sim": 0.492,
            "sds": 0.181,
        },
        {
            "configuration": "+ SFT + EMA + DPS",
            "utmos": 3.20,
            "dnsmos": 2.61,
            "wer": 8.65,
            "spk_sim": 0.508,
            "sds": 0.108,
        },
    ]


def table_ii_test_seen_speakers() -> list[dict[str, Any]]:
    """Table 2 — test set seen speakers (team leaderboard excerpt)."""
    return [
        {"team": "T01", "utmos": 3.9559, "wer": 6.48, "spk_sim": 0.2564, "adcf": "0.0453/0.1782/0.1125"},
        {"team": "T02", "utmos": 3.7390, "wer": 5.50, "spk_sim": 0.3511, "adcf": "0.0471/0.1232/0.1125"},
        {"team": "T05 (Ours)", "utmos": 3.2016, "wer": 8.65, "spk_sim": 0.2798, "adcf": "0.1582/0.5233/0.2562"},
        {"team": "T06", "utmos": 3.4909, "wer": 9.45, "spk_sim": 0.4775, "adcf": "0.1527/0.3786/0.2292"},
    ]


def headline_results() -> dict[str, Any]:
    cfg = F5TtsDpsConfig()
    return {
        "best_adcf_among_submissions": True,
        "adcf_t01": cfg.test_adcf_t01,
        "adcf_t02": cfg.test_adcf_t02,
        "adcf_t08": cfg.test_adcf_t08,
        "dev_utmos": cfg.dev_utmos,
        "dev_wer_pct": cfg.dev_wer,
        "dev_spk_sim": cfg.dev_spk_sim,
    }


def evaluation_demo(cfg: F5TtsDpsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or F5TtsDpsConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_dev_ablation": table_i_dev_ablation(),
        "table_ii_test_seen_speakers": table_ii_test_seen_speakers(),
        "headlines": headline_results(),
    }
