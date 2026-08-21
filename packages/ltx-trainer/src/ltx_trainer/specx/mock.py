"""Smoke helpers for SpecX benchmark metadata."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.specx.config import SpecxConfig
from ltx_trainer.specx.metrics import cosine_similarity, macro_f1_multilabel, top_k_accuracy
from ltx_trainer.specx.tokenize import discretize_ir_spectrum, format_1h_peak


def evaluation_smoke() -> dict[str, Any]:
    cfg = SpecxConfig()
    rng = np.random.default_rng(0)
    ir = 50 + 30 * np.sin(np.linspace(0, 4 * np.pi, 1800))
    tokens = discretize_ir_spectrum(ir)
    peak = format_1h_peak(1.24, 1.26, "t", 3.0)

    candidates = [["CCO", "CCC", "CC"], ["c1ccccc1", "CCO"]]
    truth = ["CCO", "c1ccccc1"]
    top1 = top_k_accuracy(candidates, truth, k=1)
    top5 = top_k_accuracy([c + ["X"] * 4 for c in candidates], truth, k=5)

    y_true = rng.integers(0, 2, size=(32, 8))
    y_pred = y_true.copy()
    y_pred[:4] = 1 - y_pred[:4]
    macro_f1 = macro_f1_multilabel(y_true, y_pred)

    sim = cosine_similarity(tokens.astype(np.float64), tokens.astype(np.float64))

    return {
        "total_molecules": cfg.total_molecules_filtered,
        "ir_token_len": int(tokens.size),
        "example_peak": peak,
        "top1_acc_toy": round(top1, 3),
        "top5_acc_toy": round(top5, 3),
        "macro_f1_toy": round(macro_f1, 3),
        "cosine_self": round(sim, 3),
        "paper_multimodal_top1_pct": 59.04,
    }


def pipeline_demo() -> dict[str, Any]:
    cfg = SpecxConfig()
    return {
        "tiers": {
            "Large": cfg.tier_large_molecules,
            "Small": cfg.tier_small_molecules,
            "Exp": cfg.tier_exp_molecules,
        },
        "modalities": len(cfg.modalities_all),
    }
