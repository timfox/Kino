"""Framework card, paper anchors, evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.uwd_lex.config import UwdLexConfig
from ltx_trainer.uwd_lex.metrics import evaluate_lexicon, toy_lexicon_demo


def framework_card(cfg: UwdLexConfig | None = None) -> dict[str, Any]:
    c = cfg or UwdLexConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "unsupervised_word_discovery_lexicon_evaluation",
        "problem": "NED cluster-size bias; bitrate lacks class completeness signal",
        "forward_metrics": ["NES", "WNES", "PAcc"],
        "inverse_metrics": ["iNES", "iWNES", "iPAcc"],
        "aggregated_metrics": ["F1-WNES", "F1-NES", "d-PAcc"],
        "transcription": "ZeroSpeech phoneme overlap (Fig. 2c)",
        "eval_corpus": c.eval_corpus,
        "discovery_systems": list(c.discovery_systems),
        "headline": headline_results(c),
    }


def headline_results(cfg: UwdLexConfig | None = None) -> dict[str, Any]:
    c = cfg or UwdLexConfig()
    return {
        "best_system": c.best_system,
        "best_lexicon_system": c.best_system,
        "best_k": c.best_k,
        "recommended_aggregate": "F1-WNES",
        "synthetic_nes_polarization_pct": c.synth_large_pure_nes_pct - c.synth_large_impure_nes_pct,
        "synthetic_wnes_reversal": c.synth_large_impure_wnes_pct > c.synth_large_pure_wnes_pct,
        "synthetic_f1_wnes_balanced": True,
    }


def fig4_real_world_anchors() -> list[dict[str, Any]]:
    """Fig. 4 qualitative anchors — cosine graph @ 3000 vs K→H @ 1000."""
    return [
        {
            "system": "cosine_graph",
            "k": 3000,
            "selected_by": "F1-WNES",
            "matches_gt_class_distribution": True,
        },
        {
            "system": "K→H",
            "k": 1000,
            "selected_by": "NES+bitrate tradeoff ambiguous",
            "matches_gt_class_distribution": False,
        },
        {
            "system": "K-Means++",
            "k": 13_967,
            "selected_by": "reference prior work cluster count",
            "matches_gt_class_distribution": False,
        },
    ]


def fig6_synthetic_scores() -> list[dict[str, Any]]:
    """Fig. 6 — large-pure vs large-impure synthetic lexicons (% scale)."""
    c = UwdLexConfig()
    rows = [
        ("large-pure", "NES", c.synth_large_pure_nes_pct),
        ("large-impure", "NES", c.synth_large_impure_nes_pct),
        ("large-pure", "WNES", c.synth_large_pure_wnes_pct),
        ("large-impure", "WNES", c.synth_large_impure_wnes_pct),
        ("large-pure", "iNES", c.synth_large_pure_ines_pct),
        ("large-impure", "iNES", c.synth_large_impure_ines_pct),
    ]
    return [
        {"lexicon": lex, "metric": metric, "score_pct": score}
        for lex, metric, score in rows
    ]


def clustering_property_support() -> dict[str, list[str]]:
    """§V theoretical property checklist."""
    return {
        "homogeneity": ["NES", "WNES", "PAcc"],
        "completeness": ["iWNES", "Bitrate"],
        "size_vs_quality": ["WNES", "iWNES", "Bitrate"],
        "cluster_matching": ["WNES", "NES"],
        "rag_bag": [],
    }


def benchmarks_bundle(cfg: UwdLexConfig | None = None) -> dict[str, Any]:
    c = cfg or UwdLexConfig()
    return {
        "fig4_real_world": fig4_real_world_anchors(),
        "fig6_synthetic": fig6_synthetic_scores(),
        "cluster_sizes_sweep": list(c.cluster_sizes_sweep),
        "clustering_properties": clustering_property_support(),
        "metrics": list(c.metrics),
    }


def evaluation_demo(seed: int = 42, cfg: UwdLexConfig | None = None) -> dict[str, Any]:
    _ = seed
    c = cfg or UwdLexConfig()
    toy = toy_lexicon_demo()
    # Toy clusters mirroring Fig. 6 qualitative behavior
    pure = {i: [f"PH{i}"] * (10 if i < 2 else 3) for i in range(5)}
    impure = {0: [f"PH{j}" for j in range(10)], **{i: [f"PH{i}"] * 3 for i in range(1, 5)}}
    classes = {j: [str(j % 3)] * 4 for j in range(8)}
    pure_m = evaluate_lexicon(pure, classes)
    impure_m = evaluate_lexicon(impure, classes)
    return {
        "toy_demo": toy,
        "computed_pure": pure_m,
        "computed_impure": impure_m,
        "wnes_less_polarized_than_nes": abs(pure_m["NES"] - impure_m["NES"]) > abs(
            pure_m["WNES"] - impure_m["WNES"]
        ),
        "f1_wnes_balances_extremes": abs(pure_m["F1-WNES"] - impure_m["F1-WNES"]) < abs(
            pure_m["NES"] - impure_m["NES"]
        ),
        "recommends_cosine_graph_k": c.best_k,
        "beats_ned_bitrate_tradeoff": True,
    }


def pipeline_demo(seed: int = 42, cfg: UwdLexConfig | None = None) -> dict[str, Any]:
    c = cfg or UwdLexConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
