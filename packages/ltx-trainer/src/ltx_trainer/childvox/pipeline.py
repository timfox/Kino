"""ChildVox framework card and paper tables (arXiv:2605.29257)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.childvox.config import ChildVoxCategory, ChildVoxConfig
from ltx_trainer.childvox.datasets import balanced_distribution, dataset_registry
from ltx_trainer.childvox.encoder_proxy import layer_hiddens_from_waveform
from ltx_trainer.childvox.eval_suite import eval_suite_smoke
from ltx_trainer.childvox.layout import LIMITATIONS
from ltx_trainer.childvox.models import lora_smoke, model_catalog, weighted_encoder_pool


def framework_card(cfg: ChildVoxConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ChildVoxConfig()
    return {
        "name": "ChildVox",
        "paper": cfg.paper_arxiv,
        "idea": (
            "Unified benchmark from birth through school age: physiological sounds, "
            "vocalizations, canonical syllables, and speech across 17 datasets and "
            "20+ sub-tasks. Evaluates SSL, ASR, and LALM models with LoRA fine-tuning."
        ),
        "categories": [c.value for c in ChildVoxCategory],
        "num_datasets": cfg.num_datasets,
        "num_subtasks": cfg.num_subtasks,
        "childvox_balanced": {
            "total_samples": cfg.balanced_total_samples,
            "subtasks": cfg.balanced_subtasks,
            "train_per_label": cfg.train_per_label,
        },
        "preprocessing": {
            "sample_rate_hz": cfg.sample_rate_hz,
            "min_duration_ms": cfg.min_duration_ms,
            "max_duration_s": cfg.max_duration_s,
        },
        "limitations": list(LIMITATIONS),
    }


def table_iii_encoder_macro_f1() -> list[dict[str, Any]]:
    """Table 3 excerpt — encoder Macro-F1 on full benchmark (selected rows)."""
    return [
        {"dataset": "CirCor", "SSAST": 0.636, "WavLM-L": 0.643, "Whisper-L": 0.552},
        {"dataset": "ICBHI-crackles", "SSAST": 0.644, "WavLM-L": 0.612, "Whisper-L": 0.586},
        {"dataset": "AudioSet-Child", "SSAST": 0.657, "Whisper-S": 0.634, "Whisper-L": 0.619},
        {"dataset": "SpeechMaturity", "SSAST": 0.686, "Whisper-S": 0.634, "Whisper-L": 0.640},
        {"dataset": "UltraSuite", "SSAST": 0.850, "Whisper-S": 0.926, "Whisper-L": 0.888},
        {"dataset": "C-BESD", "SSAST": 0.753, "WavLM-L": 0.892, "Whisper-L": 0.532},
        {"dataset": "SpeechOcean-Prosody", "SSAST": 0.613, "WavLM-L": 0.705, "Whisper-L": 0.715},
    ]


def table_iv_diarization_asr() -> list[dict[str, Any]]:
    """Table 4 — DER and WER (lower is better)."""
    return [
        {"model": "WavLM-Large", "NLS_DER": 22.10, "ADOS_DER": 45.60, "MyST_WER": 16.56, "ADOS_WER": 59.34},
        {"model": "Whisper-Small", "NLS_DER": 18.70, "ADOS_DER": 44.00, "MyST_WER": 15.93, "ADOS_WER": 59.91},
        {"model": "Whisper-Large", "NLS_DER": 17.70, "ADOS_DER": 42.50, "MyST_WER": 14.80, "ADOS_WER": 40.20},
        {"model": "Parakeet-TDT", "NLS_DER": None, "ADOS_DER": None, "MyST_WER": 15.82, "ADOS_WER": 45.60},
    ]


def table_v_balanced_subset() -> list[dict[str, Any]]:
    """Table 5 — ChildVox-Balanced test (Macro-F1 / WER / PER)."""
    return [
        {"model": "SSAST", "CirCor": 0.543, "AudioSet": 0.638, "SpeechMaturity": 0.735, "MyST_WER": None},
        {"model": "WavLM-Large", "CirCor": 0.571, "AudioSet": 0.538, "SpeechMaturity": 0.612, "MyST_WER": 0.147},
        {"model": "Whisper-Large", "CirCor": 0.402, "AudioSet": 0.636, "SpeechMaturity": 0.689, "MyST_WER": 0.128},
        {"model": "Qwen2-Audio", "CirCor": 0.397, "AudioSet": 0.699, "SpeechMaturity": 0.726, "MyST_WER": 0.133},
        {"model": "AudioFlamingo3", "CirCor": 0.452, "AudioSet": 0.089, "SpeechMaturity": 0.069, "MyST_WER": 0.274},
    ]


def figure_iv_proprietary_comparison() -> list[dict[str, Any]]:
    """Figure 4 — Macro-F1 vs Gemini Flash on public ChildVox-Balanced tasks."""
    return [
        {"dataset": "CirCor", "Gemini_2.5_Flash": 0.28, "Gemini_3.5_Flash": 0.31, "ChildVox_Best_Encoder": 0.571, "Qwen2-Audio": 0.397},
        {"dataset": "SPRSound", "Gemini_2.5_Flash": 0.22, "Gemini_3.5_Flash": 0.25, "ChildVox_Best_Encoder": 0.518, "Qwen2-Audio": 0.541},
        {"dataset": "ReCANVo", "Gemini_2.5_Flash": 0.18, "Gemini_3.5_Flash": 0.21, "ChildVox_Best_Encoder": 0.500, "Qwen2-Audio": 0.514},
        {"dataset": "PERCEPT-R", "Gemini_2.5_Flash": 0.45, "Gemini_3.5_Flash": 0.48, "ChildVox_Best_Encoder": 0.843, "Qwen2-Audio": 0.779},
        {"dataset": "SO-Prosody", "Gemini_2.5_Flash": 0.42, "Gemini_3.5_Flash": 0.44, "ChildVox_Best_Encoder": 0.759, "Qwen2-Audio": 0.671},
    ]


def application_language_levels() -> list[dict[str, Any]]:
    """Figure 5 — NLS utterance rate by language level (LL-1 to LL-3)."""
    return [
        {"level": "LL-1", "label": "pre-verbal", "median_utterances_per_min": 2.1},
        {"level": "LL-2", "label": "first words", "median_utterances_per_min": 4.8},
        {"level": "LL-3", "label": "word combinations", "median_utterances_per_min": 8.3},
    ]


def application_rhotic_age_correlation() -> dict[str, Any]:
    """Figure 6 — PERCEPT-R rhotic probability vs age."""
    return {"pearson_r": 0.576, "age_range_months": [115, 288], "note": "typical-developing children only"}


def headline_results() -> dict[str, Any]:
    return {
        "coverage": "17 datasets, 20+ sub-tasks, birth through school age",
        "balanced_train": "64,641 samples (ChildVox-Balanced)",
        "best_asr": "Whisper-Large 14.80 WER MyST, 40.20 WER ADOS",
        "lalm_winner": "Qwen2-Audio competitive; AudioFlamingo3 instruction-following failures",
        "vs_proprietary": "ChildVox-trained models beat Gemini 2.5/3.5 Flash on public tasks",
    }


def pipeline_demo(cfg: ChildVoxConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or ChildVoxConfig()
    rng = np.random.default_rng(seed)
    wave = rng.standard_normal(int(cfg.sample_rate_hz * 2.0)) * 0.05
    hiddens = layer_hiddens_from_waveform(wave, sample_rate=cfg.sample_rate_hz)
    clf = weighted_encoder_pool(hiddens, num_classes=5, seed=seed)
    lora = lora_smoke(89_000_000, rank=cfg.lora_rank)
    eval_out = eval_suite_smoke(cfg, seed=seed)
    return {
        "num_datasets": len(dataset_registry()),
        "balanced_total": sum(r["samples"] for r in balanced_distribution()),
        "toy_classification_pred": clf["pred"],
        "lora_rank": lora["rank"],
        "waveform_samples": int(wave.size),
        "sample_rate_hz": cfg.sample_rate_hz,
        "eval_suite": eval_out,
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "datasets": dataset_registry(),
        "balanced_distribution": balanced_distribution(),
        "model_catalog": model_catalog(),
        "table_iii_encoder_macro_f1": table_iii_encoder_macro_f1(),
        "table_iv_diarization_asr": table_iv_diarization_asr(),
        "table_v_balanced_subset": table_v_balanced_subset(),
        "figure_iv_proprietary": figure_iv_proprietary_comparison(),
        "application_language_levels": application_language_levels(),
        "application_rhotic_age": application_rhotic_age_correlation(),
        "headline": headline_results(),
        "limitations": list(LIMITATIONS),
    }
