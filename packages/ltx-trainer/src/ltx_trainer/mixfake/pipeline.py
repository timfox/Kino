"""MixFake framework card, paper tables, and smoke demos (arXiv:2605.23201)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mixfake.config import MixFakeConfig, MixFakeDatasetStats
from ltx_trainer.mixfake.layout import LIMITATIONS
from ltx_trainer.mixfake.mock import toy_feature_sequence
from ltx_trainer.mixfake.mixing import MixLabel, background_label, foreground_label
from ltx_trainer.mixfake.prompts import layer_input_width, texture_prompt
from ltx_trainer.mixfake.signals import (
    feature_flux,
    frequency_prompt_fusion,
    instantaneous_frequency_from_phase,
    teager_kaiser_sequence,
)


def framework_card(cfg: MixFakeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MixFakeConfig()
    stats = MixFakeDatasetStats()
    return {
        "name": "MixFake",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "idea": (
            "Benchmark and multi-stream prompt tuning for audio deepfake detection "
            "in mixed speech + music/environment audio: Base + HHT frequency + TKEO texture "
            "streams injected into frozen XLS-R (XLSR-AASIST)."
        ),
        "dataset": {
            "samples": stats.total_samples,
            "hours": stats.total_hours,
            "snr_db": list(cfg.snr_levels_db),
            "mix_labels": [m.value for m in MixLabel],
        },
        "streams": ["base", "frequency_hht", "texture_tkeo"],
        "backbone": cfg.backbone,
        "subtasks": ["foreground_speech_detection", "background_audio_detection"],
    }


def table_dataset_overview() -> dict[str, Any]:
    """Table I excerpt."""
    return {
        "train": {"samples": 116_000, "hours": 371.02},
        "dev": {"samples": 20_500, "hours": 43.67},
        "eval": {"samples": 116_000, "hours": 258.99},
        "total": {"samples": 252_500, "hours": 673.69},
        "per_quadrant_train": 20_000,
    }


def table_mixfake_subtasks_eer() -> dict[str, dict[str, float]]:
    """Table II — MixFake foreground/background EER (%)."""
    return {
        "XLSR-AASIST": {"foreground": 2.84, "background": 20.12},
        "XLSR-Mamba": {"foreground": 1.37, "background": 17.86},
        "WPT-XLSR-AASIST": {"foreground": 2.85, "background": 15.81},
        "OURS": {"foreground": 0.95, "background": 12.40},
    }


def table_in_the_wild_eer() -> dict[str, float]:
    """Table III — cross-dataset EER on In-the-Wild (%)."""
    return {
        "XLSR-AASIST": 9.60,
        "XLSR-Mamba": 6.71,
        "WPT-XLSR-AASIST": 7.35,
        "OURS": 6.24,
    }


def table_ablation_prompts() -> dict[str, dict[str, float]]:
    """Table IV/V — prompt stream ablation (EER %)."""
    return {
        "P_base": {"foreground": 3.05, "background": 14.31},
        "P_freq": {"foreground": 2.01, "background": 13.50},
        "P_tex": {"foreground": 2.13, "background": 14.89},
        "P_tex+P_base": {"foreground": 1.71, "background": 13.62},
        "P_freq+P_base": {"foreground": 1.50, "background": 12.86},
        "P_tex+P_freq": {"foreground": 1.35, "background": 13.10},
        "full": {"foreground": 0.95, "background": 12.40},
    }


def table_snr_highlights() -> dict[str, dict[str, float]]:
    """Fig. 2 highlighted EER points (%)."""
    return {
        "foreground": {"snr_-5db": 3.10, "snr_15db": 0.36},
        "background": {"snr_-5db": 11.24, "snr_20db": 16.70},
        "baseline_fg_snr_-5": {"XLSR-AASIST": 6.46, "WPT": 5.48},
    }


def pipeline_demo(cfg: MixFakeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MixFakeConfig()
    seq = toy_feature_sequence()
    tkeo = teager_kaiser_sequence(seq)
    flux = feature_flux(seq)
    if_val = instantaneous_frequency_from_phase(0.1, 0.35)
    f_prompt = frequency_prompt_fusion(if_val, if_val * 0.9, if_val * 0.8)
    p_tex = texture_prompt(1.0, sum(tkeo) / len(tkeo) if tkeo else 0.0)
    mix = MixLabel.FF_FB
    return {
        "mean_tkeo": sum(tkeo) / len(tkeo) if tkeo else 0.0,
        "feature_flux": flux,
        "frequency_prompt_scalar": f_prompt,
        "texture_prompt_scalar": p_tex,
        "layer_width_example": layer_input_width(n_base=64, n_freq=64, n_tex=64, n_hidden=768),
        "mix_label": mix.value,
        "fg_bona_fide": foreground_label(mix),
        "bg_bona_fide": background_label(mix),
        "snr_levels_db": list(cfg.snr_levels_db),
    }


def evaluation_demo(cfg: MixFakeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MixFakeConfig()
    sub = table_mixfake_subtasks_eer()
    bg_gain = sub["XLSR-AASIST"]["background"] - sub["OURS"]["background"]
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "background_eer_improvement_vs_xlsr_aasist": round(bg_gain, 2),
        "paper_tables": {
            "dataset": table_dataset_overview(),
            "mixfake_subtasks": sub,
            "in_the_wild": table_in_the_wild_eer(),
            "ablation": table_ablation_prompts(),
            "snr": table_snr_highlights(),
        },
    }
