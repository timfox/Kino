"""Framework card, benchmark tables, CPU demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mdd_lssg.attention import cross_attention, fuse_for_prediction
from ltx_trainer.mdd_lssg.config import MddLssgConfig
from ltx_trainer.mdd_lssg.gcn import embedding_distance, gcn_forward, phoneme_embeddings
from ltx_trainer.mdd_lssg.graph import (
    ARTICULATORY_CATEGORIES,
    build_categorical_graph,
    build_statistical_graph,
    demo_substitution_counts_spanish,
    demo_substitution_counts_vietnamese,
)


def framework_card(cfg: MddLssgConfig | None = None) -> dict[str, Any]:
    c = cfg or MddLssgConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "mispronunciation_detection_diagnosis",
        "dataset": c.dataset,
        "l1_backgrounds": list(c.l1_backgrounds),
        "audio_encoder": c.audio_encoder,
        "graph": "language_specific_statistical_confusion",
        "headline": headline_results(c),
    }


def headline_results(cfg: MddLssgConfig | None = None) -> dict[str, Any]:
    c = cfg or MddLssgConfig()
    return {
        "detection_f1": c.ours_detection_f1,
        "detection_recall": c.ours_detection_recall,
        "detection_precision": c.ours_detection_precision,
        "diagnosis_der": c.ours_der,
        "beats_cat_gcn_f1": c.ours_detection_f1 > c.cat_gcn_f1,
    }


def table1_detection(cfg: MddLssgConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or MddLssgConfig()
    return [
        {"model": "L1-aware: Aux-Embed", "recall": 0.5465, "precision": 0.5828, "f1": c.aux_embed_f1},
        {"model": "L1-aware: Look-up Embed", "recall": 0.5539, "precision": 0.5835, "f1": c.lookup_embed_f1},
        {"model": "MDDGCN", "recall": 0.6167, "precision": 0.5190, "f1": c.mddgcn_f1},
        {"model": "CAT-GCN-MDD", "recall": 0.5368, "precision": 0.6365, "f1": c.cat_gcn_f1},
        {
            "model": "MDD-LSSG (Ours)",
            "recall": c.ours_detection_recall,
            "precision": c.ours_detection_precision,
            "f1": c.ours_detection_f1,
        },
    ]


def table2_diagnosis(cfg: MddLssgConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or MddLssgConfig()
    return [
        {"model": "L1-aware: Aux-Embed", "frr": 6.47, "far": 45.35, "der": c.aux_embed_der},
        {"model": "L1-aware: Look-up Embed", "frr": 6.54, "far": 44.61, "der": c.lookup_embed_der},
        {"model": "MDDGCN", "frr": 9.18, "far": 38.03, "der": c.mddgcn_der},
        {"model": "CAT-GCN-MDD", "frr": 5.07, "far": 46.32, "der": c.cat_gcn_der},
        {"model": "MDD-LSSG (Ours)", "frr": c.ours_frr, "far": c.ours_far, "der": c.ours_der},
    ]


def table_l1_f1(cfg: MddLssgConfig | None = None) -> list[dict[str, Any]]:
    """Fig. 3 style per-L1 F1 stub."""
    c = cfg or MddLssgConfig()
    return [
        {"l1": "Arabic", "ours": 0.58, "lookup": 0.55, "cat_gcn": 0.56},
        {"l1": "Hindi", "ours": 0.57, "lookup": 0.54, "cat_gcn": 0.55},
        {"l1": "Korean", "ours": 0.59, "lookup": 0.56, "cat_gcn": 0.57},
        {"l1": "Mandarin", "ours": 0.60, "lookup": 0.57, "cat_gcn": 0.58},
        {"l1": "Spanish", "ours": c.spanish_ours_f1, "lookup": c.spanish_lookup_f1, "cat_gcn": c.spanish_cat_gcn_f1},
        {"l1": "Vietnamese", "ours": 0.61, "lookup": 0.57, "cat_gcn": 0.56},
    ]


def benchmarks_bundle(cfg: MddLssgConfig | None = None) -> dict[str, Any]:
    c = cfg or MddLssgConfig()
    return {
        "table1_detection": table1_detection(c),
        "table2_diagnosis": table2_diagnosis(c),
        "table_l1_f1": table_l1_f1(c),
        "training": {
            "optimizer": "AdamW",
            "lr": c.learning_rate,
            "batch_size": c.batch_size,
            "epochs": c.max_epochs,
            "gcn_layers": c.gcn_layers,
        },
    }


def pipeline_demo(seed: int = 42, cfg: MddLssgConfig | None = None) -> dict[str, Any]:
    c = cfg or MddLssgConfig()
    stat_vn = build_statistical_graph(demo_substitution_counts_vietnamese())
    stat_es = build_statistical_graph(demo_substitution_counts_spanish())
    cat_es = build_categorical_graph(ARTICULATORY_CATEGORIES, phonemes=stat_es.phonemes)

    z_s_weight = stat_vn.edge_weight("s", "z")
    s_z_weight = stat_vn.edge_weight("z", "s")
    assert abs(stat_vn.outgoing_sum("z") - 1.0) < 1e-6

    pair = ("d", "dh")
    d_cat, d_stat = embedding_distance(cat_es, stat_es, pair, embed_dim=c.embed_dim, seed=seed)
    stat_closer = d_stat < d_cat

    canonical = ["dh", "v", "z", "t", "s"]
    ling = phoneme_embeddings(stat_es, canonical, embed_dim=c.embed_dim, seed=seed)
    acoustic = gcn_forward(stat_es, embed_dim=c.embed_dim, seed=seed + 1)[: ling.shape[0]]
    ctx = cross_attention(acoustic, ling)
    fused = fuse_for_prediction(acoustic, ctx)

    t1 = table1_detection(c)
    ours = next(r for r in t1 if "Ours" in r["model"])
    return {
        "z_to_s_weight": round(z_s_weight, 4),
        "s_to_z_weight": round(s_z_weight, 4),
        "directional_asymmetric": z_s_weight > s_z_weight,
        "statistical_closer_confused_pair": stat_closer,
        "confused_pair": pair,
        "cat_distance": round(d_cat, 4),
        "stat_distance": round(d_stat, 4),
        "fused_dim": int(fused.shape[-1]),
        "ours_detection_f1": ours["f1"],
        "beats_baselines": ours["f1"] > c.cat_gcn_f1 and ours["f1"] > c.lookup_embed_f1,
        "ours_der": c.ours_der,
        "der_beats_mddgcn": c.ours_der < c.mddgcn_der,
    }


def evaluation_demo(seed: int = 42, cfg: MddLssgConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
