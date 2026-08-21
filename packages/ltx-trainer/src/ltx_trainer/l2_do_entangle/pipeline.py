"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.l2_do_entangle.analysis import linear_cka, stratified_cer_gaps
from ltx_trainer.l2_do_entangle.config import L2DoEntangleConfig
from ltx_trainer.l2_do_entangle.losses import cer_gap, dual_output_loss, single_output_loss


def framework_card(cfg: L2DoEntangleConfig | None = None) -> dict[str, Any]:
    c = cfg or L2DoEntangleConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "dual_output_l2_asr",
        "outputs": ["surface_transcription", "meaning_transcription"],
        "components": [
            "Shared Conformer encoder + auxiliary surface CTC",
            "Dual Transformer decoders (surface + meaning)",
            "SO baseline: separate encoders per task",
        ],
        "loss_single": "L = α LCTC + (1−α) Latt",
        "loss_dual": "L = α LCTC + β Lsurf + γ Lmean",
        "headline": headline_results(c),
    }


def headline_results(cfg: L2DoEntangleConfig | None = None) -> dict[str, Any]:
    c = cfg or L2DoEntangleConfig()
    return {
        "ko_do_meaning_cer": c.ko_do_meaning_cer,
        "ko_surface_gap": cer_gap(c.ko_do_surface_cer, c.ko_so_surface_cer),
        "ko_meaning_gap": cer_gap(c.ko_do_meaning_cer, c.ko_so_meaning_cer),
        "en_do_surface_cer": c.en_do_surface_cer,
        "en_surface_gap": cer_gap(c.en_do_surface_cer, c.en_so_surface_cer),
        "en_meaning_gap": cer_gap(c.en_do_meaning_cer, c.en_so_meaning_cer),
        "en_surface_gap_ed_gt10": c.en_surface_gap_ed_gt10,
        "en_encoder_entangled_layer11": c.en_encoder_sso_mso_layer11,
    }


def table1_dataset_stats() -> list[dict[str, Any]]:
    """Table 1 — AI-Hub L2 read-speech corpus statistics."""
    return [
        {
            "language": "Korean",
            "train": 33442,
            "val": 4180,
            "test": 4181,
            "total": 41803,
            "ed0_pct": 38.8,
            "ed1_3_pct": 35.4,
            "ed4_10_pct": 20.6,
            "ed_ge11_pct": 5.2,
        },
        {
            "language": "English",
            "train": 57616,
            "val": 7199,
            "test": 7207,
            "total": 72022,
            "ed0_pct": 26.1,
            "ed1_3_pct": 40.3,
            "ed4_10_pct": 29.5,
            "ed_ge11_pct": 4.2,
        },
    ]


def table2_cer_results(cfg: L2DoEntangleConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — Conformer / Whisper SO and Conformer DO CER (%)."""
    c = cfg or L2DoEntangleConfig()
    return [
        {
            "model": "Conformer SO",
            "params_m": c.so_conformer_params_m,
            "ko_surface": c.ko_so_surface_cer,
            "ko_meaning": c.ko_so_meaning_cer,
            "en_surface": c.en_so_surface_cer,
            "en_meaning": c.en_so_meaning_cer,
        },
        {
            "model": "Whisper-base SO",
            "params_m": 72,
            "ko_surface": 10.05,
            "ko_meaning": 4.62,
            "en_surface": 11.39,
            "en_meaning": 0.55,
        },
        {
            "model": "Whisper-small SO",
            "params_m": 244,
            "ko_surface": 6.76,
            "ko_meaning": 0.54,
            "en_surface": 11.20,
            "en_meaning": 0.27,
        },
        {
            "model": "Conformer DO",
            "params_m": c.do_params_m,
            "ko_surface": c.ko_do_surface_cer,
            "ko_meaning": c.ko_do_meaning_cer,
            "en_surface": c.en_do_surface_cer,
            "en_meaning": c.en_do_meaning_cer,
            "dual_output": True,
        },
    ]


def figure2_stratified_gaps(cfg: L2DoEntangleConfig | None = None) -> dict[str, list[dict[str, float | str]]]:
    """Figure 2 — CER gap Δ = DO − SO by edit-distance bin."""
    c = cfg or L2DoEntangleConfig()
    ko_surface = {"ED=0": 0.19, "ED=1-3": 0.45, "ED=4-10": 1.03, "ED>10": -0.12}
    ko_meaning = {"ED=0": -0.35, "ED=1-3": -0.58, "ED=4-10": -0.91, "ED>10": -1.96}
    en_surface = {
        "ED=0": c.en_surface_gap_ed0,
        "ED=1-3": 1.85,
        "ED=4-10": 3.42,
        "ED>10": c.en_surface_gap_ed_gt10,
    }
    en_meaning = {
        "ED=0": c.en_meaning_gap_ed0,
        "ED=1-3": -0.85,
        "ED=4-10": -1.72,
        "ED>10": c.en_meaning_gap_ed_gt10,
    }
    return {
        "korean": stratified_cer_gaps(surface_gaps=ko_surface, meaning_gaps=ko_meaning),
        "english": stratified_cer_gaps(surface_gaps=en_surface, meaning_gaps=en_meaning),
    }


def table3_encoder_cka() -> list[dict[str, Any]]:
    """Table 3 — layer-wise encoder CKA (selected layers)."""
    return [
        {"layer": 0, "ko_sso_mso": 0.95, "en_sso_mso": 0.91},
        {"layer": 3, "ko_sso_mso": 0.43, "en_sso_mso": 0.89},
        {"layer": 6, "ko_sso_mso": 0.56, "en_sso_mso": 0.75},
        {"layer": 9, "ko_sso_mso": 0.45, "en_sso_mso": 0.66},
        {"layer": 11, "ko_sso_mso": 0.56, "en_sso_mso": 0.40},
    ]


def table4_decoder_cka(cfg: L2DoEntangleConfig | None = None) -> list[dict[str, Any]]:
    """Table 4 — decoder CKA at layers 0, 3, 4, 7."""
    c = cfg or L2DoEntangleConfig()
    return [
        {
            "layer": 0,
            "ko_sso_mso": 0.82,
            "ko_mso_mdo": 0.84,
            "en_sso_mso": 0.34,
            "en_mso_mdo": 0.43,
        },
        {
            "layer": 3,
            "ko_sso_mso": 0.52,
            "ko_mso_mdo": 0.85,
            "en_sso_mso": 0.52,
            "en_mso_mdo": 0.59,
        },
        {
            "layer": 7,
            "ko_sso_mso": 0.53,
            "ko_mso_mdo": 0.88,
            "en_sso_mso": 0.39,
            "en_mso_mdo": c.en_decoder_mso_mdo_layer7,
            "en_mso_sdo_cross": c.en_decoder_mso_sdo_cross_layer7,
            "en_sso_mdo_cross": c.en_decoder_sso_mdo_cross_layer7,
        },
    ]


def benchmarks_bundle(cfg: L2DoEntangleConfig | None = None) -> dict[str, Any]:
    c = cfg or L2DoEntangleConfig()
    return {
        "table1_dataset_stats": table1_dataset_stats(),
        "table2_cer_results": table2_cer_results(c),
        "figure2_stratified_gaps": figure2_stratified_gaps(c),
        "table3_encoder_cka": table3_encoder_cka(),
        "table4_decoder_cka": table4_decoder_cka(c),
        "loss_weights": {
            "alpha": c.alpha_ctc,
            "beta": c.beta_surf,
            "gamma": c.gamma_mean,
        },
    }


def evaluation_demo(seed: int = 42, cfg: L2DoEntangleConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or L2DoEntangleConfig()

    ctc = float(rng.uniform(0.5, 1.5))
    att_s = float(rng.uniform(0.8, 2.0))
    att_m = float(rng.uniform(0.3, 1.2))
    l_single_s = single_output_loss(ctc, att_s, alpha=c.alpha_ctc)
    l_single_m = single_output_loss(ctc, att_m, alpha=c.alpha_ctc)
    l_dual = dual_output_loss(
        ctc,
        att_s,
        att_m,
        alpha=c.alpha_ctc,
        beta=c.beta_surf,
        gamma=c.gamma_mean,
    )

    reps_s = rng.normal(0, 1, (32, 16))
    reps_m = rng.normal(0, 1, (32, 16))
    ko_cka = linear_cka(reps_s, reps_s + rng.normal(0, 0.2, (32, 16)))
    en_cka = linear_cka(reps_s, reps_m + rng.normal(0, 0.05, (32, 16)))

    en_surf_gap = cer_gap(c.en_do_surface_cer, c.en_so_surface_cer)
    en_mean_gap = cer_gap(c.en_do_meaning_cer, c.en_so_meaning_cer)

    return {
        "l_single_surface": l_single_s,
        "l_single_meaning": l_single_m,
        "l_dual": l_dual,
        "linear_cka_ko_disentangled_stub": ko_cka,
        "linear_cka_en_entangled_stub": en_cka,
        "en_surface_degrades_under_mtl": en_surf_gap > 0,
        "en_meaning_improves_under_mtl": en_mean_gap < 0,
        "en_cross_task_inversion_layer7": c.en_decoder_mso_sdo_cross_layer7 > c.en_decoder_mso_mdo_layer7,
        "en_surface_gap": en_surf_gap,
        "en_meaning_gap": en_mean_gap,
        "en_surface_gap_scales_with_ed": c.en_surface_gap_ed_gt10 > c.en_surface_gap_ed0,
    }


def pipeline_demo(seed: int = 42, cfg: L2DoEntangleConfig | None = None) -> dict[str, Any]:
    c = cfg or L2DoEntangleConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
