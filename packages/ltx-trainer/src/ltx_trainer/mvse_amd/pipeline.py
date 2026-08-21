"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mvse_amd.config import MvseAmdConfig
from ltx_trainer.mvse_amd.features import modality_detection_features
from ltx_trainer.mvse_amd.scoring import fused_score, lambda_for_presence, max_pool_file_score


def framework_card(cfg: MvseAmdConfig | None = None) -> dict[str, Any]:
    c = cfg or MvseAmdConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "query_adaptive_av_person_retrieval",
        "corpus": "BBC Rewind (12,594 broadcast videos)",
        "components": [
            "ECAPA-TDNN speaker + ResNet-400 face embeddings (MVSE)",
            "Cross-modal score consistency for {AVP, AoP, VoP} detection",
            "Query-adaptive late fusion λ ∈ {0, 0.5, 1}",
        ],
        "headline": headline_results(c),
    }


def headline_results(cfg: MvseAmdConfig | None = None) -> dict[str, Any]:
    c = cfg or MvseAmdConfig()
    gap = c.oracle_p1 - c.fixed_p1
    recovered = c.adaptive_p1 - c.fixed_p1
    return {
        "adaptive_p1": c.adaptive_p1,
        "fixed_p1": c.fixed_p1,
        "face_p1": c.face_p1,
        "speaker_p1": c.speaker_p1,
        "oracle_p1": c.oracle_p1,
        "oracle_gap_recovery_pct": c.oracle_gap_recovery_pct,
        "detection_accuracy": c.detect_full_acc,
        "fixed_below_face_only": c.fixed_p1 < c.face_p1,
    }


def table2_modality_detection(cfg: MvseAmdConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — LoSoCV classification accuracy (%)."""
    c = cfg or MvseAmdConfig()
    return [
        {"features": "Base", "logreg": 82.3, "svm_l": 82.8, "svm_r": 82.7, "dt": 76.7},
        {"features": "+ Cross", "logreg": 88.2, "svm_l": 88.1, "svm_r": 87.9, "dt": 88.8},
        {
            "features": "+ Cross+μ+σ",
            "logreg": 88.5,
            "svm_l": 88.4,
            "svm_r": 88.2,
            "dt": c.detect_full_acc,
        },
    ]


def table3_retrieval(cfg: MvseAmdConfig | None = None) -> list[dict[str, Any]]:
    """Table 3 — P@K (%) for baselines and adaptive systems."""
    c = cfg or MvseAmdConfig()
    return [
        {"system": "Speaker", "p1": c.speaker_p1, "p3": 80.7, "p5": 78.3, "p10": 74.3},
        {"system": "Face", "p1": c.face_p1, "p3": 88.6, "p5": 86.3, "p10": 81.6},
        {"system": "Fixed", "p1": c.fixed_p1, "p3": 88.6, "p5": 87.0, "p10": 83.3},
        {
            "system": "Adaptive +Cross",
            "p1": c.adaptive_p1,
            "p3": c.adaptive_p3,
            "p5": 88.0,
            "p10": 84.1,
        },
        {"system": "Oracle", "p1": c.oracle_p1, "p3": c.oracle_p3, "p5": 89.3, "p10": 85.2},
    ]


def table4_by_presence(cfg: MvseAmdConfig | None = None) -> list[dict[str, Any]]:
    """Table 4 — P@1 (%) decomposed by presence type."""
    c = cfg or MvseAmdConfig()
    return [
        {
            "system": "Speaker",
            "avp": 86.6,
            "aop": 80.8,
            "vop": None,
        },
        {
            "system": "Face",
            "avp": 95.1,
            "aop": None,
            "vop": 93.4,
        },
        {
            "system": "Fixed",
            "avp": 93.8,
            "aop": 76.9,
            "vop": 88.5,
        },
        {
            "system": "Adaptive",
            "avp": c.adaptive_avp_p1,
            "aop": c.adaptive_aop_p1,
            "vop": c.adaptive_vop_p1,
        },
        {"system": "Oracle", "avp": 96.9, "aop": 80.8, "vop": 93.4},
    ]


def benchmarks_bundle(cfg: MvseAmdConfig | None = None) -> dict[str, Any]:
    c = cfg or MvseAmdConfig()
    return {
        "corpus": {
            "archive_videos": c.archive_videos,
            "archive_hours": c.archive_hours,
            "query_videos": c.query_videos,
            "query_speakers": c.query_speakers,
        },
        "presence_counts": {"AVP": c.avp_queries, "VoP": c.vop_queries, "AoP": c.aop_queries},
        "table2_modality_detection": table2_modality_detection(c),
        "table3_retrieval": table3_retrieval(c),
        "table4_by_presence": table4_by_presence(c),
        "fusion_lambdas": {"AVP": c.lambda_avp, "AoP": c.lambda_aop, "VoP": c.lambda_vop},
    }


def evaluation_demo(seed: int = 42, cfg: MvseAmdConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or MvseAmdConfig()
    n = c.top_n

    # Simulated AVP: peaked within-modal + high cross-modal agreement
    ss_avp = np.sort(rng.uniform(0.6, 0.95, n))[::-1]
    sf_avp = np.sort(rng.uniform(0.65, 0.96, n))[::-1]
    cs_avp = np.sort(rng.uniform(0.55, 0.90, n))[::-1]
    cf_avp = np.sort(rng.uniform(0.58, 0.88, n))[::-1]
    feat_avp = modality_detection_features(ss_avp, sf_avp, cs_avp, cf_avp)

    # Simulated VoP: low speaker scores, high face, low cross speaker→face on speaker set
    ss_vop = np.sort(rng.uniform(0.1, 0.35, n))[::-1]
    sf_vop = np.sort(rng.uniform(0.7, 0.95, n))[::-1]
    cs_vop = np.sort(rng.uniform(0.05, 0.25, n))[::-1]
    cf_vop = np.sort(rng.uniform(0.08, 0.30, n))[::-1]

    q = rng.normal(0, 1, 8)
    arch = [rng.normal(0, 1, 8) for _ in range(3)]
    spk_scores = [max_pool_file_score(q, [e]) for e in arch]
    face_scores = [max_pool_file_score(q + 0.1, [e]) for e in arch]

    fixed = fused_score(spk_scores[0], face_scores[0], lam=c.lambda_fixed)
    adaptive = fused_score(spk_scores[0], face_scores[0], lam=lambda_for_presence("VoP"))

    return {
        "feature_dim": len(feat_avp),
        "expected_feature_dim": c.feature_dim,
        "avp_cross_mean_exceeds_vop": cs_avp.mean() > cs_vop.mean(),
        "fixed_fusion_score_stub": fixed,
        "vop_adaptive_lambda": lambda_for_presence("VoP"),
        "vop_adaptive_score_stub": adaptive,
        "adaptive_beats_fixed_anchor": c.adaptive_p1 > c.fixed_p1,
        "fixed_worse_than_face_only": c.fixed_p1 < c.face_p1,
        "cross_modal_boost_pp": round(c.detect_cross_acc - c.detect_base_acc, 1),
    }


def pipeline_demo(seed: int = 42, cfg: MvseAmdConfig | None = None) -> dict[str, Any]:
    c = cfg or MvseAmdConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
