"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.speechjbb.config import SpeechJbbConfig
from ltx_trainer.speechjbb.metrics import classify_judge_label, rates_from_counts


def framework_card(cfg: SpeechJbbConfig | None = None) -> dict[str, Any]:
    c = cfg or SpeechJbbConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "lalm_code_switched_audio_jailbreak",
        "dataset": "SpeechJBB (JBB → 5 langs + 10 CS pairs + pseudo-word aug)",
        "judge": "GPT-4.1 LLM-as-a-Judge (Refused / Deflected / Jailbroken)",
        "components": [
            "XTTS synthesized monolingual + code-switched harmful audio",
            "Phonologically plausible pseudo-word insertion (10/30/50%)",
            "Nine SOTA LALMs (open + proprietary)",
        ],
        "headline": headline_results(c),
    }


def headline_results(cfg: SpeechJbbConfig | None = None) -> dict[str, Any]:
    c = cfg or SpeechJbbConfig()
    return {
        "mono_jsr": c.mono_jsr,
        "xy_jsr": c.xy_jsr,
        "jsr_gain_xy_vs_mono": round(c.xy_jsr - c.mono_jsr, 2),
        "voxtral_mean_jsr": c.voxtral_mean_jsr,
        "gemini_mean_jsr": c.gemini_mean_jsr,
        "pseudo_50_mean_jsr": c.pseudo_50_jsr,
        "pseudo_50_xy_jsr": c.pseudo_50_xy_jsr,
    }


def table1_synthesis_quality() -> list[dict[str, Any]]:
    """Table 1 — monolingual XTTS WER / UTMOS."""
    return [
        {"language": "En", "wer": 5.4, "utmos": 4.2},
        {"language": "De", "wer": 6.2, "utmos": 3.8},
        {"language": "Es", "wer": 2.4, "utmos": 3.5},
        {"language": "Fr", "wer": 7.2, "utmos": 3.4},
        {"language": "It", "wer": 4.1, "utmos": 3.4},
    ]


def table2_cs_utmos() -> list[dict[str, Any]]:
    """Table 2 — code-switched UTMOS by language pair."""
    return [
        {"pair": "En–De", "utmos": 3.8843},
        {"pair": "En–Es", "utmos": 3.7354},
        {"pair": "En–Fr", "utmos": 3.6774},
        {"pair": "En–It", "utmos": 3.6852},
        {"pair": "De–Es", "utmos": 3.7834},
        {"pair": "De–Fr", "utmos": 3.7485},
        {"pair": "De–It", "utmos": 3.7483},
        {"pair": "Es–Fr", "utmos": 3.5621},
        {"pair": "Es–It", "utmos": 3.4039},
        {"pair": "Fr–It", "utmos": 3.3182},
    ]


def table3_model_results(cfg: SpeechJbbConfig | None = None) -> list[dict[str, Any]]:
    """Table 3 — RR/DR/JSR by model and language setting."""
    c = cfg or SpeechJbbConfig()
    rows = [
        {"model": "Flamingo", "mono_rr": 66.40, "enx_jsr": 24.00, "xy_jsr": 27.83, "avg_jsr": 25.40},
        {"model": "Gemini", "mono_rr": 97.08, "enx_jsr": 2.58, "xy_jsr": 7.92, "avg_jsr": 4.76},
        {"model": "Gemma 3n", "mono_rr": 95.00, "enx_jsr": 3.75, "xy_jsr": 14.17, "avg_jsr": 8.20},
        {"model": "Gemma 4", "mono_rr": 75.00, "enx_jsr": 29.75, "xy_jsr": 33.00, "avg_jsr": 29.20},
        {"model": "GPT", "mono_rr": 93.00, "enx_jsr": 7.75, "xy_jsr": 16.83, "avg_jsr": 11.07},
        {"model": "Qwen2.5-Omni", "mono_rr": 89.40, "enx_jsr": 10.75, "xy_jsr": 15.67, "avg_jsr": 12.07},
        {"model": "Qwen3-Omni", "mono_rr": 94.60, "enx_jsr": 7.25, "xy_jsr": 12.33, "avg_jsr": 8.60},
        {"model": "SALMoNN", "mono_rr": 72.00, "enx_jsr": 19.50, "xy_jsr": 10.67, "avg_jsr": 17.73},
        {"model": "Voxtral", "mono_rr": 51.40, "enx_jsr": 47.75, "xy_jsr": 49.83, "avg_jsr": 48.27},
    ]
    rows.append(
        {
            "model": "Mean",
            "mono_rr": c.mono_rr,
            "mono_dr": c.mono_dr,
            "mono_jsr": c.mono_jsr,
            "enx_jsr": c.enx_jsr,
            "xy_jsr": c.xy_jsr,
            "avg_jsr": c.mean_jsr,
            "average": True,
        }
    )
    return rows


def table4_pseudo_word_obfuscation(cfg: SpeechJbbConfig | None = None) -> list[dict[str, Any]]:
    """Table 4 — mean JSR under pseudo-word insertion ratios."""
    c = cfg or SpeechJbbConfig()
    return [
        {"insertion": "baseline", "mean_jsr": c.mean_jsr, "xy_jsr": c.xy_jsr},
        {"insertion": "10%", "mean_jsr": c.pseudo_10_jsr, "xy_jsr": 22.32},
        {"insertion": "30%", "mean_jsr": c.pseudo_30_jsr, "xy_jsr": 24.00},
        {"insertion": "50%", "mean_jsr": c.pseudo_50_jsr, "xy_jsr": c.pseudo_50_xy_jsr},
    ]


def table6_mgsm(cfg: SpeechJbbConfig | None = None) -> list[dict[str, Any]]:
    """Table 6 — Speech-MGSM correct rate (%)."""
    c = cfg or SpeechJbbConfig()
    return [
        {"model": "Gemini", "correct": c.gemini_mgsm},
        {"model": "GPT", "correct": 91.8},
        {"model": "Voxtral", "correct": c.voxtral_mgsm},
        {"model": "Qwen3-Omni", "correct": 74.1},
        {"model": "Flamingo", "correct": 6.3},
    ]


def benchmarks_bundle(cfg: SpeechJbbConfig | None = None) -> dict[str, Any]:
    c = cfg or SpeechJbbConfig()
    return {
        "code_switch_pairs": list(c.code_switch_pairs),
        "pseudo_word_ratios": list(c.pseudo_word_ratios),
        "table1_synthesis_quality": table1_synthesis_quality(),
        "table2_cs_utmos": table2_cs_utmos(),
        "table3_model_results": table3_model_results(c),
        "table4_pseudo_word_obfuscation": table4_pseudo_word_obfuscation(c),
        "table6_mgsm": table6_mgsm(c),
    }


def evaluation_demo(seed: int = 42, cfg: SpeechJbbConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or SpeechJbbConfig()

    labels = ["Refused", "Jailbroken", "Deflected"]
    mono_counts = [
        int(rng.integers(70, 90)),
        int(rng.integers(5, 20)),
        int(rng.integers(1, 8)),
    ]
    xy_counts = [
        int(rng.integers(55, 75)),
        int(rng.integers(15, 30)),
        int(rng.integers(5, 15)),
    ]
    mono_rates = rates_from_counts(*mono_counts)
    xy_rates = rates_from_counts(*xy_counts)

    judged = [classify_judge_label(x) for x in ["Refused", "jailbroken", "Deflected", "refusal"]]
    assert judged == ["Refused", "Jailbroken", "Deflected", "Refused"]

    return {
        "synthetic_mono_rates": mono_rates,
        "synthetic_xy_rates": xy_rates,
        "xy_jsr_exceeds_mono_anchor": c.xy_jsr > c.mono_jsr,
        "voxtral_worst_anchor": c.voxtral_mean_jsr > c.gemini_mean_jsr,
        "pseudo_50_increases_jsr": c.pseudo_50_jsr > c.mean_jsr,
        "voxtral_high_mgsm_still_vulnerable": c.voxtral_mgsm > 70.0 and c.voxtral_mean_jsr > 40.0,
        "n_code_switch_pairs": len(c.code_switch_pairs),
        "n_models": c.n_models,
    }


def pipeline_demo(seed: int = 42, cfg: SpeechJbbConfig | None = None) -> dict[str, Any]:
    c = cfg or SpeechJbbConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
