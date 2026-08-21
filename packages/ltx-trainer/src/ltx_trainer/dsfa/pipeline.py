"""Framework card and benchmark tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dsfa.augmentation import dsfa_demo
from ltx_trainer.dsfa.config import DsfaConfig
from ltx_trainer.dsfa.domain_gap import proxy_wild_gap_demo
from ltx_trainer.dsfa.losses import loss_demo


def framework_card(cfg: DsfaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DsfaConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "framework": cfg.framework,
        "backbone": cfg.backbone,
        "components": [
            "post_trained_ssl_backbone",
            "domain_shift_feature_augmentation",
            "cosg_exteval_benchmark",
        ],
        "proxy_data": "CoRS (CodecFake+)",
        "wild_eval": ["CoSG Eval", "CoSG ExtEval"],
        "headline": {
            "cosg_eval_eer_pct": cfg.cosg_eval_eer_k,
            "cosg_exteval_eer_pct": cfg.cosg_exteval_eer_k,
        },
    }


def table1_cosg_datasets() -> list[dict[str, Any]]:
    """Table 1 — CoSG Eval vs CoSG ExtEval statistics."""
    return [
        {
            "eval_set": "CoSG Eval",
            "bona_fide_samples": 850,
            "spoof_samples": 931,
            "spoof_models": 17,
            "duration_h": 1.32,
            "mean_duration_s": 5.51,
        },
        {
            "eval_set": "CoSG ExtEval",
            "bona_fide_samples": 1222,
            "spoof_samples": 1366,
            "spoof_models": 40,
            "duration_h": 3.08,
            "mean_duration_s": 8.13,
        },
    ]


def table2_main_results() -> list[dict[str, Any]]:
    """Table 2 — selected CodecFake+ EER rows (%)."""
    return [
        {"model": "(f) CoRS DEC Balance", "backbone": "Wav2Vec2-AASIST", "cosg_eval": 11.91, "cosg_exteval": 27.07},
        {"model": "(g) PT-Wav2Vec2", "backbone": "PT-Wav2Vec2", "cosg_eval": 3.95, "cosg_exteval": 22.19},
        {"model": "(h) PT-Wav2Vec2-FT", "backbone": "PT-Wav2Vec2-FT", "cosg_eval": 3.56, "cosg_exteval": 22.19},
        {"model": "(i) + SupCon", "backbone": "PT-Wav2Vec2-FT", "cosg_eval": 3.00, "cosg_exteval": 24.08},
        {"model": "(j) + DSFA + SupCon", "backbone": "PT-Wav2Vec2-FT", "cosg_eval": 2.78, "cosg_exteval": 23.00},
        {"model": "(k) + DSFA", "backbone": "PT-Wav2Vec2-FT", "cosg_eval": 3.00, "cosg_exteval": 21.80},
    ]


def table3_layer_ablation() -> list[dict[str, Any]]:
    """Table 3 — layer-wise DSFA distribution ablation."""
    return [
        {"layer": 1, "uniform_eval": 3.12, "uniform_exteval": 23.11, "gaussian_eval": 2.78, "gaussian_exteval": 22.61},
        {"layer": 6, "uniform_eval": 3.12, "uniform_exteval": 23.00, "gaussian_eval": 3.00, "gaussian_exteval": 23.58},
        {"layer": 12, "uniform_eval": 3.00, "uniform_exteval": 23.50, "gaussian_eval": 3.00, "gaussian_exteval": 24.00},
        {"layer": 18, "uniform_eval": 3.00, "uniform_exteval": 23.93, "gaussian_eval": 3.45, "gaussian_exteval": 24.58},
        {"layer": 24, "uniform_eval": 2.78, "uniform_exteval": 22.85, "gaussian_eval": 2.78, "gaussian_exteval": 23.00},
    ]


def table4_dsfa_probability() -> list[dict[str, Any]]:
    """Table 4 — DSFA probability ablation."""
    return [
        {"probability": 0.00, "cosg_eval": 3.00, "cosg_exteval": 24.08},
        {"probability": 0.25, "cosg_eval": 2.78, "cosg_exteval": 22.77},
        {"probability": 0.50, "cosg_eval": 2.78, "cosg_exteval": 23.00},
        {"probability": 0.75, "cosg_eval": 2.78, "cosg_exteval": 23.08},
        {"probability": 1.00, "cosg_eval": 2.78, "cosg_exteval": 24.00},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_cosg_datasets": table1_cosg_datasets(),
        "table2_main_results": table2_main_results(),
        "table3_layer_ablation": table3_layer_ablation(),
        "table4_dsfa_probability": table4_dsfa_probability(),
    }


def headline_results(cfg: DsfaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DsfaConfig()
    return {
        "cosg_eval_eer_pct": cfg.cosg_eval_eer_k,
        "cosg_exteval_eer_pct": cfg.cosg_exteval_eer_k,
        "exteval_gain_vs_dec_f_pct": round(cfg.cosg_exteval_eer_f - cfg.cosg_exteval_eer_k, 2),
        "cosg_exteval_spoof_models": cfg.cosg_exteval_spoof_models,
    }


def evaluation_demo(*, seed: int = 0, cfg: DsfaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DsfaConfig()
    return {
        "framework": framework_card(cfg),
        "dsfa": dsfa_demo(seed=seed, cfg=cfg),
        "losses": loss_demo(seed=seed, cfg=cfg),
        "domain_gap": proxy_wild_gap_demo(seed=seed, cfg=cfg),
        "headline": headline_results(cfg),
    }
