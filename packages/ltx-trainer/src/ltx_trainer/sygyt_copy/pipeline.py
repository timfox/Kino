"""Framework card, Table 2/3/4 anchors, CPU demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sygyt_copy.config import SygytCopyConfig
from ltx_trainer.sygyt_copy.losses import combined_loss, overtone_salience, overtone_salience_loss
from ltx_trainer.sygyt_copy.metrics import formant_peak_error_hz, lsd_reduction_pct
from ltx_trainer.sygyt_copy.waveguide import clip_damping, reflection_coefficient, three_way_scattering


def framework_card(cfg: SygytCopyConfig | None = None) -> dict[str, Any]:
    c = cfg or SygytCopyConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "venue": c.venue,
        "modalities": ["audio"],
        "style": "sygyt biphonic singing",
        "companion": c.companion_url,
    }


def table2_copy_synthesis(cfg: SygytCopyConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or SygytCopyConfig()
    return [
        {
            "dataset": "HFA",
            "method": "Artic. chain",
            "dofs": c.articulator_dofs + 6,
            "lsd_db": c.hfa_artic_lsd,
            "spcorr": 0.71,
        },
        {
            "dataset": "HFA",
            "method": "DDSP",
            "dofs": 100_000,
            "lsd_db": c.hfa_ddsp_lsd,
            "spcorr": 0.82,
        },
        {
            "dataset": "HFA",
            "method": "B-spline",
            "dofs": c.bspline_dofs + 16,
            "lsd_db": c.hfa_bspline_lsd,
            "spcorr": 0.86,
        },
        {
            "dataset": "Bergevin",
            "method": "Artic. chain",
            "dofs": c.articulator_dofs + 6,
            "lsd_db": c.berg_artic_lsd,
            "spcorr": 0.66,
        },
        {
            "dataset": "Bergevin",
            "method": "DDSP",
            "dofs": 100_000,
            "lsd_db": c.berg_ddsp_lsd,
            "spcorr": 0.83,
        },
        {
            "dataset": "Bergevin",
            "method": "B-spline",
            "dofs": c.bspline_dofs + 16,
            "lsd_db": c.berg_bspline_lsd,
            "spcorr": 0.88,
        },
    ]


def table3_overtone_errors(cfg: SygytCopyConfig | None = None) -> dict[str, Any]:
    c = cfg or SygytCopyConfig()
    return {
        "bspline": {
            "delta_er": c.bspline_delta_er,
            "delta_sot_db": c.bspline_delta_sot_db,
            "delta_hpr": c.bspline_delta_hpr,
        },
        "artic_chain": {"delta_er": 0.36, "delta_sot_db": 2.18, "delta_hpr": 10.84},
        "ddsp": {"delta_er": 0.35, "delta_sot_db": 1.70, "delta_hpr": 12.43},
    }


def table4_ablation(cfg: SygytCopyConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or SygytCopyConfig()
    return [
        {"condition": "Full", "sublingual": True, "damping": True, "lsd_db": c.ablation_full_lsd},
        {
            "condition": "No sublingual",
            "sublingual": False,
            "damping": True,
            "lsd_db": c.ablation_no_sublingual_lsd,
        },
        {
            "condition": "No damping",
            "sublingual": True,
            "damping": False,
            "lsd_db": c.ablation_no_damping_lsd,
        },
        {
            "condition": "Minimal",
            "sublingual": False,
            "damping": False,
            "lsd_db": c.ablation_minimal_lsd,
        },
    ]


def benchmarks_bundle(cfg: SygytCopyConfig | None = None) -> dict[str, Any]:
    c = cfg or SygytCopyConfig()
    return {
        "table2": table2_copy_synthesis(c),
        "table3": table3_overtone_errors(c),
        "table4": table4_ablation(c),
        "formant_peak_error_hz": {
            "bspline": c.formant_peak_error_bspline_hz,
            "artic": c.formant_peak_error_artic_hz,
            "ddsp": c.formant_peak_error_ddsp_hz,
        },
    }


def pipeline_demo(seed: int = 42, cfg: SygytCopyConfig | None = None) -> dict[str, Any]:
    c = cfg or SygytCopyConfig()
    _ = seed

    hfa_red = lsd_reduction_pct(c.hfa_artic_lsd, c.hfa_bspline_lsd)
    berg_red = lsd_reduction_pct(c.berg_artic_lsd, c.berg_bspline_lsd)

    r = reflection_coefficient(2.0, 1.5)
    d = clip_damping(1.05)
    scatter = three_way_scattering(2.0, 1.8, 0.9)

    sot = overtone_salience(100.0, 10.0)
    lot = overtone_salience_loss(sot, sot + 0.5)
    loss = combined_loss(1.0, 0.5, 0.3, 0.2, lot)

    peak_err = formant_peak_error_hz(2100.0, 2100.0 + c.formant_peak_error_bspline_hz)

    ablation = table4_ablation(c)
    sublingual_delta = ablation[1]["lsd_db"] - ablation[0]["lsd_db"]
    damping_delta = ablation[2]["lsd_db"] - ablation[0]["lsd_db"]

    bspline_beats_ddsp_hfa = c.hfa_bspline_lsd < c.hfa_ddsp_lsd
    bspline_beats_artic_all = all(
        row["method"] == "B-spline" and row["lsd_db"]
        < next(r["lsd_db"] for r in table2_copy_synthesis(c) if r["dataset"] == row["dataset"] and r["method"] == "Artic. chain")
        for row in table2_copy_synthesis(c)
        if row["method"] == "B-spline"
    )

    return {
        "segments": c.segments,
        "lsd_reduction_hfa_pct": round(hfa_red, 1),
        "lsd_reduction_berg_pct": round(berg_red, 1),
        "hfa_reduction_matches_paper": 29.0 <= hfa_red <= 31.0,
        "berg_reduction_matches_paper": 37.0 <= berg_red <= 39.0,
        "reflection_coefficient": round(r, 4),
        "damping_clipped": d == 0.9999,
        "scatter_sum_finite": all(abs(x) <= 1.0 for x in scatter),
        "overtone_salience_db": round(sot, 2),
        "combined_loss": round(loss, 3),
        "formant_peak_error_hz": peak_err,
        "sublingual_ablation_lsd_delta_db": round(sublingual_delta, 2),
        "damping_ablation_lsd_delta_db": round(damping_delta, 2),
        "sublingual_dominates_ablation": sublingual_delta > damping_delta,
        "bspline_beats_ddsp_hfa": bspline_beats_ddsp_hfa,
        "bspline_beats_artic_all_datasets": bspline_beats_artic_all,
        "diphonic_frame_pct": c.diphonic_frame_pct,
        "overtone_band_hz": list(c.overtone_band_hz),
    }


def evaluation_demo(seed: int = 42, cfg: SygytCopyConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
