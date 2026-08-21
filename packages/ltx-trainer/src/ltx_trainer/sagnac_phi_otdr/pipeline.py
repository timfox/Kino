"""Framework card, Table 2/4, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sagnac_phi_otdr.alignment import optimal_lag
from ltx_trainer.sagnac_phi_otdr.config import SagnacPhiOtdrConfig
from ltx_trainer.sagnac_phi_otdr.grouping import BEST_GROUPING, DEFAULT_CONTIGUOUS
from ltx_trainer.sagnac_phi_otdr.metrics import accuracy, macro_f1, nar_fnr


def framework_card(cfg: SagnacPhiOtdrConfig | None = None) -> dict[str, Any]:
    c = cfg or SagnacPhiOtdrConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "das_six_class_event_recognition",
        "sensing": f"{c.sensing_distance_km:g}-km ϕ-OTDR + Sagnac auxiliary phase",
        "input": f"{c.n_channels}-channel matrix, dual-branch via grouping",
        "metrics": ["accuracy", "macro-F1", "NAR", "FNR", "latency"],
        "github": c.github_repo,
        "headline": headline_results(c),
    }


def headline_results(cfg: SagnacPhiOtdrConfig | None = None) -> dict[str, Any]:
    c = cfg or SagnacPhiOtdrConfig()
    return {
        "fusion_accuracy": c.fusion_accuracy,
        "fusion_macro_f1": c.fusion_macro_f1,
        "fusion_nar": c.fusion_nar,
        "fusion_fnr": c.fusion_fnr,
        "branch_b_cnn_accuracy": c.branch_b_cnn_accuracy,
        "best_grouping_accuracy": c.best_grouping_accuracy,
        "n_classes": c.n_classes,
        "total_samples": c.total_samples,
    }


def table2_benchmark(cfg: SagnacPhiOtdrConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — representative method comparison."""
    c = cfg or SagnacPhiOtdrConfig()
    return [
        {
            "method": "STFT + SVM",
            "accuracy": c.stft_svm_accuracy,
            "macro_f1": c.stft_svm_macro_f1,
            "nar": 36.25,
            "fnr": 22.00,
            "latency_ms": 0.0122,
        },
        {
            "method": "MPE + ZCR + SVM",
            "accuracy": c.mpe_zcr_svm_accuracy,
            "macro_f1": 43.35,
            "nar": 47.50,
            "fnr": 10.75,
            "latency_ms": 0.0127,
        },
        {
            "method": "Fusion Features + PSVM",
            "accuracy": c.psvm_accuracy,
            "macro_f1": 57.40,
            "nar": 32.50,
            "fnr": 3.25,
            "latency_ms": 0.0173,
        },
        {
            "method": "Branch B CNN",
            "accuracy": c.branch_b_cnn_accuracy,
            "macro_f1": c.branch_b_cnn_macro_f1,
            "nar": 12.50,
            "fnr": 0.00,
            "latency_ms": 3.3474,
        },
        {
            "method": "Fusion CNN",
            "accuracy": c.fusion_accuracy,
            "macro_f1": c.fusion_macro_f1,
            "nar": c.fusion_nar,
            "fnr": c.fusion_fnr,
            "latency_ms": c.fusion_latency_ms,
        },
    ]


def table4_grouping_search(cfg: SagnacPhiOtdrConfig | None = None) -> list[dict[str, Any]]:
    """Table 4 — representative channel grouping candidates."""
    c = cfg or SagnacPhiOtdrConfig()
    return [
        {
            "rank": 1,
            "branch_a": list(BEST_GROUPING.branch_a),
            "branch_b": list(BEST_GROUPING.branch_b),
            "accuracy": c.best_grouping_accuracy,
            "macro_f1": 77.46,
            "nar": 0.00,
            "fnr": 2.00,
        },
        {
            "rank": 5,
            "branch_a": list(DEFAULT_CONTIGUOUS.branch_a),
            "branch_b": list(DEFAULT_CONTIGUOUS.branch_b),
            "accuracy": c.default_split_accuracy,
            "macro_f1": 48.22,
            "nar": 0.00,
            "fnr": 44.50,
        },
    ]


def benchmarks_bundle(cfg: SagnacPhiOtdrConfig | None = None) -> dict[str, Any]:
    c = cfg or SagnacPhiOtdrConfig()
    return {
        "table2_methods": table2_benchmark(c),
        "table4_grouping": table4_grouping_search(c),
        "event_classes": list(c.event_classes),
        "dataset": {
            "total": c.total_samples,
            "train": c.train_samples,
            "test": c.test_samples,
        },
    }


def pipeline_demo(seed: int = 42, cfg: SagnacPhiOtdrConfig | None = None) -> dict[str, Any]:
    """CPU stub: cross-correlation alignment + grouping validation."""
    c = cfg or SagnacPhiOtdrConfig()
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal(128)
    xp = np.roll(xs, 3) + rng.standard_normal(128) * 0.05
    lag, peak = optimal_lag(xs, xp)
    BEST_GROUPING.validate(c.n_channels)
    DEFAULT_CONTIGUOUS.validate(c.n_channels)
    fusion = next(r for r in table2_benchmark(c) if r["method"] == "Fusion CNN")
    shallow = next(r for r in table2_benchmark(c) if "STFT" in r["method"])
    return {
        "alignment_lag": lag,
        "alignment_peak": round(peak, 4),
        "best_grouping": BEST_GROUPING.name,
        "grouping_gain_vs_default": round(c.best_grouping_accuracy - c.default_split_accuracy, 2),
        "fusion_accuracy": fusion["accuracy"],
        "shallow_accuracy": shallow["accuracy"],
        "fusion_nar": fusion["nar"],
        "fusion_fnr": fusion["fnr"],
    }


def evaluation_demo(seed: int = 42, cfg: SagnacPhiOtdrConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
