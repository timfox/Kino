"""Cause-aware SDS framework card and paper tables (arXiv:2605.25404)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cause_sds.config import CauseSdsConfig
from ltx_trainer.cause_sds.layout import LIMITATIONS
from ltx_trainer.cause_sds.mock import evaluation_smoke


def framework_card(cfg: CauseSdsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CauseSdsConfig()
    return {
        "name": "Proactive for Uncertainty — Cause-Aware SDS",
        "paper": cfg.paper_arxiv,
        "authors": "Peng, Ma, Liu, Chao, Chen, Chng (NTU / SJTU)",
        "paradigm": "Cascaded ASR-LLM with cause-aware detectors + K-round clarification",
        "asr_backbone": cfg.asr_backbone,
        "detectors": [
            "Comprehension (token, joint embedding)",
            "Perception (token, joint embedding)",
            "Deletion (frame, encoder embedding)",
            "Distortion Event (6-class, encoder-aligned)",
        ],
        "fusion_priority": "Comprehension > Perception > Deletion",
        "error_tags": ["<noise>", "<unknown>", "<del>"],
        "baseline": f"Tsallis entropy (α={cfg.tsallis_alpha})",
        "clarification": {
            "rounds": cfg.clarification_rounds,
            "dialogue_llm": cfg.llm_dialogue,
            "user_sim_llm": cfg.llm_user_sim,
            "tts": cfg.tts,
        },
        "detector_arch": f"{cfg.detector_cnn_layers}-layer 1D-CNN (~{cfg.detector_params_m}M params each)",
        "limitations": LIMITATIONS,
    }


def table_i_token_detection() -> list[dict[str, Any]]:
    """Table 1 — token-level deletion / perception / comprehension (excerpts)."""
    return [
        {
            "task": "Perception (avg)",
            "deletion_fpr": 0.77,
            "deletion_recall": 56.36,
            "perc_fpr": 5.81,
            "perc_recall": 67.53,
        },
        {
            "task": "Comprehension AESRC-Test",
            "deletion_fpr": None,
            "deletion_recall": None,
            "perc_fpr": 1.61,
            "perc_recall": 47.98,
        },
        {
            "task": "Comprehension SPGI2-Test",
            "deletion_fpr": None,
            "deletion_recall": None,
            "perc_fpr": 1.78,
            "perc_recall": 70.26,
        },
    ]


def table_ii_word_detection() -> list[dict[str, Any]]:
    """Table 2 — word-level vs Tsallis baseline (key rows)."""
    return [
        {
            "condition": "Perception avg (proposed)",
            "fpr": 9.64,
            "recall": 52.26,
            "baseline_recall": 41.57,
        },
        {
            "condition": "Comprehension SPGI2-Test",
            "fpr": 3.98,
            "recall": 57.96,
            "baseline_recall": 23.66,
        },
        {
            "condition": "Comprehension AESRC-Test",
            "fpr": 1.13,
            "recall": 39.38,
            "baseline_recall": 13.26,
        },
        {
            "condition": "OOD Alpaca",
            "fpr": 4.65,
            "recall": 71.05,
            "baseline_recall": 53.33,
        },
        {
            "condition": "OOD wsj-eval92",
            "fpr": 0.69,
            "recall": 29.72,
            "baseline_recall": 7.37,
        },
    ]


def table_iii_distortion_events() -> list[dict[str, Any]]:
    """Table 3 — distortion event detector (macro metrics)."""
    return [
        {"class": "Clean", "f1": 91.89, "accuracy": 88.92},
        {"class": "Noisy", "f1": 15.79, "accuracy": 35.85},
        {"class": "RIR", "f1": 6.76, "accuracy": 9.59},
        {"class": "Interference", "f1": 29.26, "accuracy": 38.79},
        {"class": "Packet Loss", "f1": 26.32, "accuracy": 25.44},
        {"class": "Missing", "f1": 78.58, "accuracy": 73.67},
        {"class": "Average (macro)", "f1": 41.43, "accuracy": 85.49},
    ]


def table_iv_wer_clarification() -> list[dict[str, Any]]:
    """Table 4 — 3-round interactive clarification WER."""
    return [
        {"dataset": "WSJ-eval92", "init": 4.23, "step1": 4.02, "step2": 3.90, "step3": 3.85},
        {"dataset": "Gigaspeech", "init": 14.57, "step1": 12.88, "step2": 12.47, "step3": 12.28},
        {"dataset": "SPGI-noise", "init": 17.57, "step1": 15.24, "step2": 13.85, "step3": 12.31},
        {"dataset": "AESRC-Indian", "init": 6.08, "step1": 4.91, "step2": 4.79, "step3": 4.74},
    ]


def table_v_dialogue_maj() -> list[dict[str, Any]]:
    """Table 5 — downstream MaJ on OpenHermes / Alpaca."""
    return [
        {"input": "GT Oracle", "openhermes": 86.6, "alpaca": 84.8},
        {"input": "ASR Clean", "openhermes": 85.4, "alpaca": 83.6},
        {"input": "ASR Distorted", "openhermes": 74.6, "alpaca": 68.8},
        {"input": "Clarified 3-round", "openhermes": 83.0, "alpaca": 80.8},
    ]


def headline_results() -> dict[str, Any]:
    t2 = table_ii_word_detection()
    spgi = next(r for r in t2 if "SPGI2" in r["condition"])
    t4 = table_iv_wer_clarification()
    wsj = next(r for r in t4 if r["dataset"] == "WSJ-eval92")
    t5 = table_v_dialogue_maj()
    dist = next(r for r in t5 if r["input"] == "ASR Distorted")
    clar = next(r for r in t5 if r["input"] == "Clarified 3-round")
    return {
        "finding": (
            "Cause-aware detectors more than double comprehension recall on domain shift "
            "(57.96% vs 23.66% on SPGI2). K-round clarification cuts WER up to ~30% and "
            "restores MaJ toward clean-ASR upper bounds."
        ),
        "comprehension_recall_gain_spgi2": round(spgi["recall"] - spgi["baseline_recall"], 2),
        "wsj_wer_initial": wsj["init"],
        "wsj_wer_final": wsj["step3"],
        "maj_openhermes_recovery": round(clar["openhermes"] - dist["openhermes"], 1),
        "maj_alpaca_recovery": round(clar["alpaca"] - dist["alpaca"], 1),
        "werr_spgi_noise_pct": 29.9,
        "werr_aesrc_indian_pct": 22.0,
    }


def evaluation_demo(cfg: CauseSdsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CauseSdsConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_token_detection": table_i_token_detection(),
        "table_ii_word_detection": table_ii_word_detection(),
        "table_iii_distortion_events": table_iii_distortion_events(),
        "table_iv_wer_clarification": table_iv_wer_clarification(),
        "table_v_dialogue_maj": table_v_dialogue_maj(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
