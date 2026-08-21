"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.film_spk_asr.config import FilmSpkAsrConfig
from ltx_trainer.film_spk_asr.film import (
    film_generator,
    gate_alpha,
    gated_film_modulate,
    identity_init_gamma_beta,
    mask_speaker_embedding,
)


def framework_card(cfg: FilmSpkAsrConfig | None = None) -> dict[str, Any]:
    c = cfg or FilmSpkAsrConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "pathological_asr",
        "speech_llm": c.speech_llm,
        "speaker_model": c.speaker_model,
        "trainable_params_m": c.trainable_params_m,
        "trainable_fraction_pct": c.trainable_fraction_pct,
        "components": [
            "Frozen Voxtral-Mini encoder + connector + LLM",
            "SiAmResNet34 x-vector extractor (fine-tuned)",
            "Per-layer FiLM generators (gamma, beta, gate alpha)",
            "Normative-speech embedding mask (identity on healthy speech)",
        ],
        "headline": headline_results(c),
    }


def headline_results(cfg: FilmSpkAsrConfig | None = None) -> dict[str, Any]:
    c = cfg or FilmSpkAsrConfig()
    return {
        "spkcond_torgo_wer_pp": c.spkcond_torgo_wer_pp,
        "spkcond_neurovoz_wer": c.spkcond_neurovoz_wer,
        "spkcond_mcqa_sex": c.mcqa_spkcond_sex,
        "beats_base_mcqa_sex": c.mcqa_spkcond_sex > c.mcqa_base_sex,
        "trainable_fraction_pct": c.trainable_fraction_pct,
        "flora_mcqa_collapsed": c.mcqa_flora_sex < c.mcqa_random_sex,
    }


def table1_asr_wer(cfg: FilmSpkAsrConfig | None = None) -> list[dict[str, Any]]:
    """Table 1 — WER (%) on NeuroVoz and TORGO."""
    c = cfg or FilmSpkAsrConfig()
    return [
        {
            "model": "Base",
            "trained_blocks": "None",
            "neurovoz_wer": c.base_neurovoz_wer,
            "torgo_wer": c.base_torgo_wer,
            "torgo_single_word": c.base_torgo_single_word,
            "torgo_multi_word": c.base_torgo_multi_word,
        },
        {
            "model": "FFT",
            "trained_blocks": "Encoder, Connector, Decoder",
            "neurovoz_wer": c.fft_neurovoz_wer,
            "torgo_wer": c.fft_torgo_wer,
        },
        {
            "model": "F-LoRA",
            "trained_blocks": "Encoder, Connector, Decoder",
            "neurovoz_wer": c.flora_neurovoz_wer,
            "torgo_wer": c.flora_torgo_wer,
            "best_neurovoz": True,
        },
        {
            "model": "Spk-Cond",
            "trained_blocks": "SiAmResNet34 + FiLM generators",
            "neurovoz_wer": c.spkcond_neurovoz_wer,
            "torgo_wer": c.spkcond_torgo_wer,
            "torgo_wer_pp": c.spkcond_torgo_wer_pp,
            "torgo_multi_word_pp": c.spkcond_torgo_multi_word_pp,
            "best_efficiency": True,
        },
    ]


def table2_mcqa(cfg: FilmSpkAsrConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — MCQA accuracy (%)."""
    c = cfg or FilmSpkAsrConfig()
    return [
        {"model": "Random", "overall": 33.1, "sex": c.mcqa_random_sex, "age": 25.0},
        {"model": "Majority class", "overall": None, "sex": c.mcqa_majority_sex, "age": 69.7},
        {"model": "Base", "overall": 52.7, "sex": c.mcqa_base_sex, "age": 24.8},
        {"model": "FFT", "overall": 60.9, "sex": c.mcqa_fft_sex, "age": 21.8},
        {"model": "EFT", "overall": 63.5, "sex": c.mcqa_eft_sex, "age": 16.4, "best_sex": True},
        {"model": "F-LoRA", "overall": 8.4, "sex": c.mcqa_flora_sex, "age": 1.2},
        {"model": "Spk-Cond", "overall": 59.3, "sex": c.mcqa_spkcond_sex, "age": 11.5},
    ]


def benchmarks_bundle(cfg: FilmSpkAsrConfig | None = None) -> dict[str, Any]:
    c = cfg or FilmSpkAsrConfig()
    return {
        "table1_asr_wer": table1_asr_wer(c),
        "table2_mcqa": table2_mcqa(c),
        "datasets": {
            "torgo_hours": c.torgo_hours,
            "neurovoz_hours": c.neurovoz_hours,
            "mcqa_pairs": c.mcqa_pairs,
        },
    }


def evaluation_demo(seed: int = 42, cfg: FilmSpkAsrConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or FilmSpkAsrConfig()
    d = 64
    seq = 20

    z_path = rng.normal(0, 1, c.xvector_dim)
    z_norm = mask_speaker_embedding(z_path, is_normative=True)
    assert np.allclose(z_norm, 0.0)

    w1 = rng.normal(0, 0.01, (32, c.xvector_dim))
    b1 = np.zeros(32)
    w2 = rng.normal(0, 0.001, (2 * d, 32))
    b2 = np.concatenate([np.ones(d), np.zeros(d)])
    gamma, beta = film_generator(z_path, w1, b1, w2, b2)

    wg = rng.normal(0, 0.01, c.xvector_dim)
    bg = np.array([c.gate_bias_init])
    alpha = gate_alpha(z_path, wg, bg)

    hidden = rng.normal(0, 0.1, (seq, d))
    modulated = gated_film_modulate(hidden, gamma, beta, alpha)
    g0, b0 = identity_init_gamma_beta(d)
    identity_out = gated_film_modulate(hidden, g0, b0, alpha=0.0)

    spk = next(r for r in table1_asr_wer(c) if r["model"] == "Spk-Cond")
    eft = next(r for r in table2_mcqa(c) if r.get("best_sex"))

    return {
        "normative_embedding_zero": bool(np.allclose(z_norm, 0.0)),
        "modulated_shape": list(modulated.shape),
        "identity_preserves_hidden": np.allclose(identity_out, hidden),
        "gate_alpha": alpha,
        "spkcond_torgo_pp_gain": c.base_torgo_wer - c.spkcond_torgo_wer_pp,
        "spkcond_beats_base_mcqa_sex": c.mcqa_spkcond_sex > c.mcqa_base_sex,
        "eft_best_sex_accuracy": eft["sex"],
        "spkcond_trainable_fraction_pct": c.trainable_fraction_pct,
        "flora_wer_best_neurovoz": spk["neurovoz_wer"] > c.flora_neurovoz_wer,
    }


def pipeline_demo(seed: int = 42, cfg: FilmSpkAsrConfig | None = None) -> dict[str, Any]:
    c = cfg or FilmSpkAsrConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
