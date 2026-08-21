"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.child_asr_age_adapter.adapters import (
    age_router_logits,
    bottleneck_adapter,
    combine_encoder_representations,
    film_modulate,
    route_top_k,
    transducer_nll,
)
from ltx_trainer.child_asr_age_adapter.config import ChildAsrAgeAdapterConfig


def framework_card(cfg: ChildAsrAgeAdapterConfig | None = None) -> dict[str, Any]:
    c = cfg or ChildAsrAgeAdapterConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "author": c.author,
        "task": "children_asr",
        "backbone": c.backbone,
        "challenge": c.challenge,
        "age_groups": list(c.age_groups),
        "components": [
            "Frozen Parakeet-tdt-0.6B FastConformer backbone",
            "Shared child bottleneck adapter (db=128)",
            "Age-specialized adapter bank (db=32 × 4 groups)",
            "Lightweight age router on layer-4 adapter features",
            "Unified age-conditioned FiLM adapter (comparison)",
        ],
        "headline": headline_results(c),
    }


def headline_results(cfg: ChildAsrAgeAdapterConfig | None = None) -> dict[str, Any]:
    c = cfg or ChildAsrAgeAdapterConfig()
    return {
        "shared_wer": c.shared_wer,
        "best_wer": c.best_wer,
        "best_macro_wer": c.best_macro_wer,
        "wer_improvement_vs_shared": round(c.shared_wer - c.best_wer, 1),
        "macro_improvement_vs_shared": round(c.shared_macro_wer - c.best_macro_wer, 1),
        "pt_top2_matches_gt": c.pt_top2_wer == c.best_wer,
        "age_specialized_beats_film": c.best_wer < c.film_gt_hom_wer,
        "router_accuracy": c.router_accuracy,
    }


def table1_dataset_stats() -> list[dict[str, Any]]:
    """Table 1 — speakers / hours by age group."""
    return [
        {"split": "train", "3_4_unknown": "200/27", "5_7": "1629/72", "8_11": "1012/180", "12_plus": "148/6", "unknown": "66/3.1", "total": "2993/288"},
        {"split": "dev", "3_4_unknown": "11/1.6", "5_7": "52/3.0", "8_11": "35/5.4", "12_plus": "5/0.3", "unknown": "4/0.2", "total": "101/10.5"},
        {"split": "test", "3_4_unknown": "13/1.9", "5_7": "133/5.2", "8_11": "80/12.1", "12_plus": "17/0.8", "unknown": "11/0.4", "total": "250/20.3"},
    ]


def table2_wer_results(cfg: ChildAsrAgeAdapterConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — WER (%) on Word Track test set."""
    c = cfg or ChildAsrAgeAdapterConfig()
    return [
        {"method": "Freeze backbone", "train_age": "-", "infer_age": "-", "wer": c.freeze_wer, "macro_wer": c.freeze_macro_wer},
        {"method": "Child shared (db=128)", "train_age": "-", "infer_age": "-", "wer": c.shared_wer, "macro_wer": c.shared_macro_wer},
        {"method": "Age-specialized only", "train_age": "GT", "infer_age": "GT", "wer": c.age_only_wer, "macro_wer": c.age_only_macro_wer},
        {"method": "Child shared + age-specialized", "train_age": "GT", "infer_age": "GT", "wer": c.best_wer, "macro_wer": c.best_macro_wer, "best": True},
        {"method": "Child shared + age-specialized", "train_age": "GT", "infer_age": "PT top-1", "wer": c.pt_top1_wer, "macro_wer": c.pt_top1_macro_wer},
        {"method": "Child shared + age-specialized", "train_age": "GT", "infer_age": "PT top-2", "wer": c.pt_top2_wer, "macro_wer": c.pt_top2_macro_wer},
        {"method": "Child shared + stacked adapter", "train_age": "-", "infer_age": "-", "wer": c.stacked_wer, "macro_wer": c.stacked_macro_wer},
        {"method": "Child shared + FiLM (hom., GT)", "train_age": "GT", "infer_age": "GT", "wer": c.film_gt_hom_wer, "macro_wer": c.film_gt_hom_macro_wer},
        {"method": "Child shared + FiLM (hom., PT all)", "train_age": "PT all", "infer_age": "PT all", "wer": c.film_pt_hom_wer, "macro_wer": c.film_pt_hom_macro_wer},
    ]


def table2_group_wer(cfg: ChildAsrAgeAdapterConfig | None = None) -> dict[str, float]:
    """Group-specific WER for best GT-routed model."""
    c = cfg or ChildAsrAgeAdapterConfig()
    return {
        "3_4": c.best_wer_3_4,
        "5_7": c.best_wer_5_7,
        "8_11": c.best_wer_8_11,
        "12_plus": c.best_wer_12_plus,
        "unknown": c.best_wer_unknown,
    }


def benchmarks_bundle(cfg: ChildAsrAgeAdapterConfig | None = None) -> dict[str, Any]:
    c = cfg or ChildAsrAgeAdapterConfig()
    return {
        "table1_dataset_stats": table1_dataset_stats(),
        "table2_wer_results": table2_wer_results(c),
        "table2_group_wer_best": table2_group_wer(c),
        "router_accuracy": c.router_accuracy,
        "router_macro_f1": c.router_macro_f1,
    }


def evaluation_demo(seed: int = 42, cfg: ChildAsrAgeAdapterConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or ChildAsrAgeAdapterConfig()
    d, db = c.hidden_dim, c.child_bottleneck
    seq = 16

    hidden = rng.normal(0, 0.1, (seq, d))
    down = rng.normal(0, 0.01, (db, d))
    up = rng.normal(0, 0.01, (d, db))
    adapted = bottleneck_adapter(hidden, down, up)

    pooled = adapted.mean(axis=0)
    w1 = rng.normal(0, 0.01, (c.router_hidden, d))
    b1 = np.zeros(c.router_hidden)
    w2 = rng.normal(0, 0.01, (len(c.age_groups), c.router_hidden))
    b2 = np.zeros(len(c.age_groups))
    logits = age_router_logits(pooled, w1, b1, w2, b2)
    posteriors = np.exp(logits - logits.max())
    posteriors /= posteriors.sum()
    top_idx, weights = route_top_k(posteriors, k=2)
    reps = [rng.normal(0, 0.1, d) for _ in top_idx]
    fused = combine_encoder_representations(reps, weights)

    z = rng.normal(0, 0.1, (seq, c.film_bottleneck))
    age_emb = rng.normal(0, 0.1, c.film_age_embed_dim)
    gamma_w = rng.normal(0, 0.01, (c.film_age_embed_dim, c.film_bottleneck))
    beta_w = np.zeros((c.film_age_embed_dim, c.film_bottleneck))
    film_out = film_modulate(z, age_emb, gamma_w, beta_w, c.film_gate_init)

    log_probs = rng.random((4, 50))
    log_probs /= log_probs.sum(axis=-1, keepdims=True)
    targets = np.zeros_like(log_probs)
    targets[np.arange(4), rng.integers(0, 50, 4)] = 1.0
    loss = transducer_nll(log_probs, targets)

    rows = table2_wer_results(c)
    best = next(r for r in rows if r.get("best"))
    shared = next(r for r in rows if r["method"] == "Child shared (db=128)")
    film = next(r for r in rows if "FiLM (hom., GT)" in r["method"])

    return {
        "adapted_shape": list(adapted.shape),
        "router_top2_groups": top_idx,
        "fused_encoder_dim": int(fused.shape[0]),
        "film_output_shape": list(film_out.shape),
        "transducer_nll": loss,
        "best_wer": best["wer"],
        "shared_wer": shared["wer"],
        "age_specialized_beats_shared": best["wer"] < shared["wer"],
        "age_specialized_beats_film": best["wer"] < film["wer"],
        "pt_top2_matches_gt_wer": c.pt_top2_wer == c.best_wer,
        "router_accuracy": c.router_accuracy,
    }


def pipeline_demo(seed: int = 42, cfg: ChildAsrAgeAdapterConfig | None = None) -> dict[str, Any]:
    c = cfg or ChildAsrAgeAdapterConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
