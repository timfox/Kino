"""Framework card, Table 2 anchors, CPU demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clinical_scribe_stress.config import ClinicalScribeStressConfig
from ltx_trainer.clinical_scribe_stress.metrics import (
    err_prop,
    ins_rate,
    neg_err_rate,
    triage_match,
    unsafe_rate,
    wer,
)


def framework_card(cfg: ClinicalScribeStressConfig | None = None) -> dict[str, Any]:
    c = cfg or ClinicalScribeStressConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "paired_acoustic_stress_test",
        "pipeline": f"{c.asr_model} → {c.llm_model}",
        "design": "within-encounter paired noise; frozen downstream LLM",
        "noise_taxonomy": [
            "stationary_ambient (DEMAND)",
            "non_stationary_semantic (MUSAN speech)",
        ],
        "headline": headline_results(c),
    }


def headline_results(cfg: ClinicalScribeStressConfig | None = None) -> dict[str, Any]:
    c = cfg or ClinicalScribeStressConfig()
    return {
        "clean_wer": c.clean_wer,
        "ambient_15db_wer_delta_pp": c.ambient_15db_wer_delta,
        "clean_unsafe_pct": c.clean_unsafe,
        "ambient_15db_unsafe_pct": c.ambient_15db_unsafe,
        "unsafe_near_double": c.ambient_15db_unsafe_ratio >= 1.9,
        "semantic_5db_unsafe_pct": c.semantic_5db_unsafe,
        "n_encounters": c.n_encounters,
    }


def table2_main_results(cfg: ClinicalScribeStressConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — paired acoustic stress test main results."""
    c = cfg or ClinicalScribeStressConfig()
    return [
        {
            "condition": "Reference (clean-ASR)",
            "wer": c.clean_wer,
            "ins_rate": c.clean_ins_rate,
            "neg_err": c.clean_neg_err,
            "err_prop": None,
            "triage_match": 100.0,
            "scer": 0.0,
            "under_triage": 0.0,
            "mean_score": c.clean_mean_score,
            "unsafe": c.clean_unsafe,
        },
        {
            "condition": "MUSAN Speech 15 dB",
            "wer": 21.16,
            "ins_rate": 4.74,
            "neg_err": 34.93,
            "err_prop": 68.49,
            "triage_match": 91.91,
            "scer": 75.37,
            "under_triage": 2.21,
            "mean_score": 3.92,
            "unsafe": 66.91,
        },
        {
            "condition": "MUSAN Speech 5 dB",
            "wer": c.semantic_5db_wer,
            "ins_rate": 14.55,
            "neg_err": 51.10,
            "err_prop": 10.53,
            "triage_match": 81.99,
            "scer": 92.28,
            "under_triage": 7.72,
            "mean_score": 3.41,
            "unsafe": c.semantic_5db_unsafe,
        },
        {
            "condition": "DEMAND Ambient 15 dB",
            "wer": c.ambient_15db_wer,
            "ins_rate": 2.48,
            "neg_err": c.ambient_15db_neg_err,
            "err_prop": c.ambient_15db_err_prop,
            "triage_match": 93.01,
            "scer": 44.12,
            "under_triage": 3.31,
            "mean_score": 4.45,
            "unsafe": c.ambient_15db_unsafe,
        },
        {
            "condition": "DEMAND Ambient 5 dB",
            "wer": 20.63,
            "ins_rate": 2.31,
            "neg_err": 39.71,
            "err_prop": 77.93,
            "triage_match": 91.54,
            "scer": 66.18,
            "under_triage": 3.31,
            "mean_score": 4.23,
            "unsafe": c.ambient_5db_unsafe,
        },
        {
            "condition": "Mitigation Semantic 5 dB",
            "wer": c.semantic_5db_wer,
            "ins_rate": 14.55,
            "neg_err": 51.10,
            "err_prop": 10.62,
            "triage_match": 86.03,
            "scer": 83.82,
            "under_triage": 2.57,
            "mean_score": 3.71,
            "unsafe": c.mitig_semantic_5db_unsafe,
        },
        {
            "condition": "Mitigation Ambient 5 dB",
            "wer": 20.63,
            "ins_rate": 2.31,
            "neg_err": 39.71,
            "err_prop": 78.82,
            "triage_match": 92.28,
            "scer": 51.84,
            "under_triage": 1.10,
            "mean_score": 4.36,
            "unsafe": c.mitig_ambient_5db_unsafe,
        },
    ]


def benchmarks_bundle(cfg: ClinicalScribeStressConfig | None = None) -> dict[str, Any]:
    c = cfg or ClinicalScribeStressConfig()
    return {
        "table2_main": table2_main_results(c),
        "cause_side_metrics": ["WER", "InsRate", "NegErr", "ErrProp"],
        "result_side_metrics": ["TriageMatch", "SCER", "UnderTriage", "MeanScore", "Unsafe"],
        "snr_levels_db": list(c.snr_levels_db),
    }


def pipeline_demo(seed: int = 42, cfg: ClinicalScribeStressConfig | None = None) -> dict[str, Any]:
    """CPU stub: micro WER + paired triage/unsafe on synthetic dialogue batch."""
    c = cfg or ClinicalScribeStressConfig()
    rng_seed = seed
    # Deterministic toy batch mirroring paper scale fraction
    n = 8
    ref_words = 120
    clean_sub, clean_del, clean_ins = 18, 2, 4
    noisy_sub, noisy_del, noisy_ins = 19, 2, 4  # +0.71pp WER-like delta at scale

    clean_wer = wer(clean_sub, clean_del, clean_ins, ref_words * n)
    noisy_wer = wer(noisy_sub, noisy_del, noisy_ins, ref_words * n)

    clean_triage = [2, 1, 2, 3, 2, 1, 2, 2]
    noisy_triage = [2, 1, 2, 3, 2, 2, 2, 2]  # one drift
    clean_scores = [5, 4, 5, 4, 5, 3, 4, 5]
    noisy_scores = [5, 4, 2, 4, 5, 3, 4, 5]  # one unsafe

    return {
        "seed": rng_seed,
        "n_dialogues_demo": n,
        "clean_wer_demo": round(clean_wer, 2),
        "noisy_wer_demo": round(noisy_wer, 2),
        "wer_delta_demo_pp": round(noisy_wer - clean_wer, 2),
        "triage_match_demo": round(100.0 * triage_match(clean_triage, noisy_triage), 2),
        "clean_unsafe_demo": round(unsafe_rate(clean_scores), 2),
        "noisy_unsafe_demo": round(unsafe_rate(noisy_scores), 2),
        "err_prop_demo": err_prop(delta_claim_errors=3, delta_asr_errors=1),
        "ins_rate_demo": round(ins_rate(clean_ins, ref_words * n), 2),
        "neg_err_demo": neg_err_rate(negation_mismatches=2, n_dialogues=n),
        "paper_wer_delta_15db": c.ambient_15db_wer_delta,
        "paper_unsafe_ratio_15db": c.ambient_15db_unsafe_ratio,
    }


def evaluation_demo(seed: int = 42, cfg: ClinicalScribeStressConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(seed=seed, cfg=cfg)
    c = cfg or ClinicalScribeStressConfig()
    return {**demo, "n_encounters": c.n_encounters}
