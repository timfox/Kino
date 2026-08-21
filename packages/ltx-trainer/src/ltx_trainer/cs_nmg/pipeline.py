"""Framework card, Tables 2–3, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cs_nmg.config import CsNmgConfig
from ltx_trainer.cs_nmg.losses import losses_demo
from ltx_trainer.cs_nmg.near_miss import near_miss_demo
from ltx_trainer.cs_nmg.poi import poi_demo


def framework_card(cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    c = cfg or CsNmgConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "backbone": c.backbone,
        "components": [
            "poi_detection_pier",
            "n_best_near_miss_seeding",
            "llm_poi_expansion_offline",
            "tri_level_filter_gate",
            "poi_weighted_wce_anchor",
            "infonce_contrastive_ranking",
        ],
        "datasets": list(c.datasets),
        "llm_offline": c.llm,
        "headline": headline_results(c),
    }


def table2_main_results() -> list[dict[str, Any]]:
    """Table 2 — CS-FLEURS cmn-eng and ViMedCSS vie-eng."""
    return [
        {"method": "CE", "cmn_wer": 16.67, "cmn_pier": 17.25, "vie_wer": 24.72, "vie_pier": 21.95},
        {"method": "WCE", "cmn_wer": 16.42, "cmn_pier": 16.68, "vie_wer": 24.21, "vie_pier": 21.18},
        {"method": "MWER", "cmn_wer": 15.75, "cmn_pier": 16.41, "vie_wer": 23.82, "vie_pier": 20.84},
        {"method": "CE + CL (N-best NM)", "cmn_wer": 15.64, "cmn_pier": 16.21, "vie_wer": 23.16, "vie_pier": 20.11},
        {"method": "WCE + CL (N-best NM)", "cmn_wer": 14.93, "cmn_pier": 15.72, "vie_wer": 22.86, "vie_pier": 19.10},
        {"method": "WCE + CL (tri-level)", "cmn_wer": 14.06, "cmn_pier": 15.10, "vie_wer": 21.87, "vie_pier": 18.74},
    ]


def table3_filter_ablations() -> list[dict[str, Any]]:
    """Table 3 — NM source and gating ablations."""
    return [
        {"variant": "N-best", "group": "No filter", "cmn_wer": 14.93, "cmn_pier": 15.72, "cmn_nm_utt": 1.40, "vie_wer": 22.86, "vie_pier": 19.10, "vie_nm_utt": 1.22},
        {"variant": "N-best + LLM", "group": "No filter", "cmn_wer": 15.06, "cmn_pier": 15.28, "cmn_nm_utt": 6.00, "vie_wer": 22.19, "vie_pier": 19.65, "vie_nm_utt": 6.00},
        {"variant": "N-best + LLM", "group": "Ac. only", "cmn_wer": 15.55, "cmn_pier": 15.18, "cmn_nm_utt": 5.75, "vie_wer": 24.03, "vie_pier": 19.56, "vie_nm_utt": 4.93},
        {"variant": "N-best + LLM", "group": "Ac. + Ph.", "cmn_wer": 14.50, "cmn_pier": 15.16, "cmn_nm_utt": 5.65, "vie_wer": 23.17, "vie_pier": 19.73, "vie_nm_utt": 4.92},
        {"variant": "N-best + LLM", "group": "Ac. + Text", "cmn_wer": 14.12, "cmn_pier": 14.69, "cmn_nm_utt": 3.76, "vie_wer": 24.03, "vie_pier": 19.71, "vie_nm_utt": 3.82},
        {"variant": "N-best + LLM", "group": "Ac.+Ph.+Text", "cmn_wer": 14.06, "cmn_pier": 15.10, "cmn_nm_utt": 3.77, "vie_wer": 21.87, "vie_pier": 18.74, "vie_nm_utt": 3.81},
    ]


def training_protocol(cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    c = cfg or CsNmgConfig()
    return {
        "backbone": c.backbone,
        "n_best": c.n_best,
        "near_miss_k": c.near_miss_k,
        "lambda_cl": c.lambda_cl,
        "acoustic_margin_delta": c.acoustic_margin_delta,
        "tau_text": c.tau_text,
        "tau_phoneme": c.tau_phoneme,
        "alpha_wce_cmn": c.alpha_wce_cmn,
        "alpha_wce_vie": c.alpha_wce_vie,
        "infonce_beta": c.infonce_beta,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table2_main_results": table2_main_results(),
        "table3_filter_ablations": table3_filter_ablations(),
        "training_protocol": training_protocol(),
    }


def headline_results(cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    c = cfg or CsNmgConfig()
    return {
        "cmn_wer": c.cmn_wer,
        "cmn_pier": c.cmn_pier,
        "vie_wer": c.vie_wer,
        "vie_pier": c.vie_pier,
        "wer_gain_cmn_pct": round(c.cmn_wer_ce - c.cmn_wer, 2),
        "pier_gain_cmn_pct": round(c.cmn_pier_ce - c.cmn_pier, 2),
        "wer_gain_vie_pct": round(c.vie_wer_ce - c.vie_wer, 2),
        "pier_gain_vie_pct": round(c.vie_pier_ce - c.vie_pier, 2),
    }


def evaluation_demo(*, seed: int = 0, cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    c = cfg or CsNmgConfig()
    return {
        "framework": framework_card(c),
        "poi": poi_demo(cfg=c),
        "near_miss": near_miss_demo(cfg=c),
        "losses": losses_demo(seed=seed, cfg=c),
        "headline": headline_results(c),
    }
