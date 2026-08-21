"""Framework card and benchmark tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.biear.binaural import binaural_demo
from ltx_trainer.biear.config import BiearConfig
from ltx_trainer.biear.controller import controller_demo
from ltx_trainer.biear.filterbank import filterbank_demo
from ltx_trainer.biear.sad_net import sad_net_demo


def framework_card(cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "framework": cfg.framework,
        "github": cfg.github,
        "inspiration": "MOC efferent Q-factor modulation",
        "components": [
            "erb_gabor_adaptive_filterbank",
            "dual_neural_q_controllers",
            "ild_ipd_cc_binaural_cues",
            "eight_sector_sad_nets",
        ],
        "n_sectors": cfg.n_sectors,
        "n_subbands": cfg.n_subbands,
        "best_variant": "BiEAR + Dual Controller + Rel.",
        "headline": {
            "azim_mae_1spk_seen_deg": cfg.azim_mae_1spk_seen,
            "meeting_detect_transfer_pct": cfg.meeting_detect_transfer,
        },
    }


def table1_brir_datasets() -> list[dict[str, Any]]:
    """Table 1 — BRIR datasets and azimuth/distance ranges."""
    return [
        {"environment": "Anechoic", "brir": "Anechoic", "azimuth_deg": "0–360", "distance_m": "0.5/1/2/3"},
        {"environment": "Meeting Room", "brir": "Spirit", "azimuth_deg": "90–270", "distance_m": "2"},
        {"environment": "Lecture Hall", "brir": "Auditorium3", "azimuth_deg": "90–270", "distance_m": "1.5/2.93/3.97/5.49"},
    ]


def table2_anechoic() -> list[dict[str, Any]]:
    """Table 2 — anechoic performance (selected rows)."""
    return [
        {
            "model": "DeepEar",
            "params_m": 2.08,
            "1spk_azim_mae_seen": 0.80,
            "1spk_dist_seen": 95.03,
            "3spk_detect_seen": 89.23,
        },
        {
            "model": "AuralNet",
            "params_m": 1.37,
            "1spk_azim_mae_seen": 0.73,
            "1spk_dist_seen": 98.12,
            "3spk_detect_seen": 89.80,
        },
        {
            "model": "BiEAR w/o Controller",
            "params_m": 1.29,
            "1spk_azim_mae_seen": 0.63,
            "1spk_dist_seen": 96.73,
            "3spk_detect_seen": 86.91,
        },
        {
            "model": "BiEAR + Dual Controller + Rel.",
            "params_m": 1.63,
            "1spk_azim_mae_seen": 0.36,
            "1spk_azim_mae_unseen": 0.39,
            "1spk_dist_seen": 97.84,
            "2spk_azim_mae_seen": 3.05,
            "2spk_dist_seen": 86.61,
            "3spk_detect_seen": 90.82,
            "3spk_azim_mae_seen": 8.03,
            "3spk_dist_seen": 73.91,
        },
    ]


def table3_real_rooms() -> list[dict[str, Any]]:
    """Table 3 — practical environments (BiEAR + Dual + Rel + transfer)."""
    return [
        {
            "room": "Meeting Room",
            "model": "BiEAR + Dual Controller + Rel.",
            "transfer": False,
            "1spk_detect": 70.39,
            "1spk_azim_mae": 12.31,
            "1spk_dist": 69.73,
        },
        {
            "room": "Meeting Room",
            "model": "BiEAR + Dual Controller + Rel.",
            "transfer": True,
            "1spk_detect": 93.74,
            "1spk_azim_mae": 3.92,
            "1spk_dist": 93.98,
        },
        {
            "room": "Lecture Hall",
            "model": "BiEAR + Dual Controller + Rel.",
            "transfer": True,
            "1spk_detect": 93.22,
            "1spk_azim_mae": 3.35,
            "1spk_dist": 90.89,
        },
        {
            "room": "Meeting Room",
            "model": "AuralNet + env. transfer",
            "transfer": True,
            "1spk_detect": 88.31,
            "1spk_azim_mae": 5.19,
            "1spk_dist": 92.19,
        },
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_brir_datasets": table1_brir_datasets(),
        "table2_anechoic": table2_anechoic(),
        "table3_real_rooms": table3_real_rooms(),
    }


def headline_results(cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    return {
        "azim_mae_1spk_seen_deg": cfg.azim_mae_1spk_seen,
        "dist_acc_1spk_seen_pct": cfg.dist_acc_1spk_seen,
        "meeting_detect_transfer_pct": cfg.meeting_detect_transfer,
        "gain_vs_deepear_azim_deg": round(cfg.deepear_azim_mae_1spk - cfg.azim_mae_1spk_seen, 2),
    }


def evaluation_demo(*, seed: int = 0, cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    return {
        "framework": framework_card(cfg),
        "filterbank": filterbank_demo(seed=seed, cfg=cfg),
        "controller": controller_demo(seed=seed, cfg=cfg),
        "binaural": binaural_demo(seed=seed, cfg=cfg),
        "sad_net": sad_net_demo(seed=seed, cfg=cfg),
        "headline": headline_results(cfg),
    }
