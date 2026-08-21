"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.costa.augmentation import (
    augmentation_factor_curve,
    build_text_prompt,
    optimal_augmentation_factor,
    tta_probability_average,
)
from ltx_trainer.costa.config import CostaConfig


def _ratio_numerator(ratio: str) -> int:
    """Parse ``28/37`` style beat counts for ordering."""
    return int(ratio.split("/", 1)[0])


def framework_card(cfg: CostaConfig | None = None) -> dict[str, Any]:
    c = cfg or CostaConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "speech_alzheimers_detection_tts_augmentation",
        "dataset": c.dataset,
        "components": [
            "CS-Cond CosyVoice2 (instruction fine-tuning)",
            "CS-Cond F5-TTS (cognition label + flow matching)",
            "MT + 36 ASR transcript pool",
            "Self-reference 2× + intra-class cross-synthesis",
            "WavLM AD detector + TTA probability averaging",
        ],
        "transcript_pool": c.transcript_pool_size,
        "tts_backends": list(c.tts_backends),
        "headline": headline_results(c),
    }


def headline_results(cfg: CostaConfig | None = None) -> dict[str, Any]:
    c = cfg or CostaConfig()
    return {
        "baseline_accuracy_pct": c.baseline_accuracy_pct,
        "best_accuracy_pct": c.best_accuracy_pct,
        "gain_over_baseline_pct": c.gain_over_baseline_pct,
        "best_config": "CS-Cond CosyVoice2 + fine-tuned w2v960-large-lv ASR + TTA",
        "optimal_aug_factor": c.optimal_aug_factor,
    }


def table1_tts_objective() -> list[dict[str, Any]]:
    """Table 1 — TTS test-set objective metrics (selected rows)."""
    return [
        {
            "tts_model": "CosyVoice2",
            "variant": "Pretrained-AD",
            "mcd": 6.854,
            "log_f0_rmse": 0.328,
            "fad": 8.542,
        },
        {
            "tts_model": "CosyVoice2",
            "variant": "CS-Cond-AD",
            "mcd": 5.436,
            "log_f0_rmse": 0.305,
            "fad": 2.192,
        },
        {
            "tts_model": "CosyVoice2",
            "variant": "CS-Cond-HC",
            "mcd": 6.933,
            "log_f0_rmse": 0.301,
            "fad": 2.459,
        },
        {
            "tts_model": "F5-TTS",
            "variant": "CS-Cond-AD",
            "mcd": 5.734,
            "log_f0_rmse": 0.310,
            "fad": 2.545,
        },
    ]


def table2_headline_configs() -> list[dict[str, Any]]:
    """Table 2 — blue-bold CS-Cond CosyVoice2 configs (>84%)."""
    c = CostaConfig()
    return [
        {
            "tts_model": "CS-Cond CosyVoice2",
            "text_source": "fine-tuned w2v960",
            "accuracy_pct": c.cs_cosy_w2v960_ft_pct,
        },
        {
            "tts_model": "CS-Cond CosyVoice2",
            "text_source": "fine-tuned w2v960 large lv",
            "accuracy_pct": c.cs_cosy_w2v960_large_lv_ft_pct,
        },
        {
            "tts_model": "CS-Cond CosyVoice2",
            "text_source": "fine-tuned whisper large v3",
            "accuracy_pct": c.cs_cosy_whisper_large_v3_ft_pct,
        },
        {
            "tts_model": "CS-Cond CosyVoice2",
            "text_source": "Manual transcripts (MT)",
            "accuracy_pct": 82.50,
        },
    ]


def table3_tta() -> list[dict[str, Any]]:
    """Table 3 — test-time augmentation @ 2×."""
    c = CostaConfig()
    rows = [
        ("fine-tuned w2v960", c.cs_cosy_w2v960_ft_pct, c.tta_w2v960_pct),
        ("fine-tuned w2v960 large lv", c.cs_cosy_w2v960_large_lv_ft_pct, c.tta_w2v960_large_lv_pct),
        ("fine-tuned whisper large v3", c.cs_cosy_whisper_large_v3_ft_pct, c.tta_whisper_large_v3_pct),
    ]
    return [
        {
            "text_source": src,
            "without_tta_pct": wo,
            "with_tta_pct": wt,
            "delta_pct": round(wt - wo, 2),
        }
        for src, wo, wt in rows
    ]


def table4_comparisons() -> list[dict[str, Any]]:
    """Table 4 — traditional DA + prior studies."""
    c = CostaConfig()
    return [
        {"method": "Baseline (WavLM-based)", "accuracy_pct": c.baseline_accuracy_pct},
        {"method": "+ Noise Addition", "accuracy_pct": 82.50},
        {"method": "+ Pitch Shifting", "accuracy_pct": c.pitch_shift_da_pct},
        {"method": "Whisper + MLP [33]", "accuracy_pct": c.whisper_mlp_prior_pct},
        {"method": "Wav2Vec2 + Linear [34]", "accuracy_pct": c.wav2vec2_linear_prior_pct},
        {"method": "AW-HuBERT [35]", "accuracy_pct": c.aw_hubert_prior_pct},
        {"method": "CoSTA (Ours)", "accuracy_pct": c.best_accuracy_pct},
    ]


def table2_beat_ratios() -> dict[str, str]:
    c = CostaConfig()
    return {
        "cs_cosy_beat_baseline": c.cs_cosy_beat_baseline_ratio,
        "pretrained_cosy_beat_baseline": c.pretrained_cosy_beat_baseline_ratio,
        "cs_f5_beat_baseline": c.cs_f5_beat_baseline_ratio,
        "cs_cosy_asr_beats_mt": c.cs_cosy_asr_beats_mt_ratio,
        "cs_f5_asr_beats_mt": c.cs_f5_asr_beats_mt_ratio,
    }


def benchmarks_bundle(cfg: CostaConfig | None = None) -> dict[str, Any]:
    c = cfg or CostaConfig()
    return {
        "table1_tts_objective": table1_tts_objective(),
        "table2_headline_configs": table2_headline_configs(),
        "table2_beat_ratios": table2_beat_ratios(),
        "table3_tta": table3_tta(),
        "table4_comparisons": table4_comparisons(),
        "fig3_aug_factor_curve": augmentation_factor_curve(),
        "asr_wer_span_pct": [c.asr_wer_min_pct, c.asr_wer_max_pct],
    }


def evaluation_demo(seed: int = 42, cfg: CostaConfig | None = None) -> dict[str, Any]:
    _ = seed
    c = cfg or CostaConfig()
    prompt_ad = build_text_prompt("AD", "The boy is taking cookies from the jar.")
    p_ori = (0.35, 0.65)
    p_syn = (0.42, 0.58)
    p_final = tta_probability_average(p_ori, p_syn)
    curve = augmentation_factor_curve()
    opt = optimal_augmentation_factor()
    best_row = max(table2_headline_configs(), key=lambda r: r["accuracy_pct"])
    return {
        "cs_cond_prompt_ad_prefix": prompt_ad.split("<|endofprompt|>")[0],
        "tta_probabilities": {"original": p_ori, "synthetic": p_syn, "final": p_final},
        "predicts_ad_after_tta": p_final[1] > p_final[0],
        "optimal_aug_factor": opt,
        "aug_curve_peak_accuracy": max(curve["average"]),
        "beats_baseline": c.best_accuracy_pct > c.baseline_accuracy_pct,
        "best_table2_config": best_row,
        "cs_cond_beats_pretrained": _ratio_numerator(c.cs_cosy_beat_baseline_ratio)
        > _ratio_numerator(c.pretrained_cosy_beat_baseline_ratio),
    }


def pipeline_demo(seed: int = 42, cfg: CostaConfig | None = None) -> dict[str, Any]:
    c = cfg or CostaConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
