"""Ethical pluralism analysis — entropy, confidence, bridge theories (arXiv:2605.28707)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from ltx_trainer.ethical_pluralism.config import EthicalPluralismConfig
from ltx_trainer.ethical_pluralism.ensemble import (
    StackedEnsembleModel,
    build_feature_matrix,
    evaluate_classifier,
)
from ltx_trainer.ethical_pluralism.features import EthicalCase
from ltx_trainer.ethical_pluralism.features import infer_normative_scores
from ltx_trainer.ethical_pluralism.simplex import normative_entropy
from ltx_trainer.ethical_pluralism.taxonomy import SUBTHEORY_BY_ID


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return probs
    z = np.log(probs + 1e-12) / temperature
    return np.exp(z) / np.exp(z).sum()


def classification_entropy(probs: np.ndarray) -> float:
    ent = 0.0
    for p in probs:
        if p > 1e-12:
            ent -= p * math.log(p)
    return ent


def confidence_stratification(
    model: StackedEnsembleModel,
    cases: list[EthicalCase],
    cfg: EthicalPluralismConfig | None = None,
    *,
    bins: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 1.0),
) -> list[dict[str, Any]]:
    """Figure 5 style — accuracy vs top-1 confidence."""
    cfg = cfg or EthicalPluralismConfig()
    buckets: dict[str, list[bool]] = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": [] for i in range(len(bins) - 1)}
    X = build_feature_matrix(
        cases,
        cfg,
        use_normative=model.feature_flags.get("normative", True),
        use_context=model.feature_flags.get("context", True),
        use_embeddings=model.feature_flags.get("embeddings", True),
    )
    for c, x in zip(cases, X):
        p = model.predict_proba_row(x)
        p = temperature_scale_probs(p, cfg.temperature_scaling)
        conf = float(np.max(p))
        ok = model.predict(x) == c.subtheory_id
        for i in range(len(bins) - 1):
            if bins[i] <= conf < bins[i + 1] or (i == len(bins) - 2 and conf == bins[i + 1]):
                buckets[f"{bins[i]:.1f}-{bins[i+1]:.1f}"].append(ok)
                break
    rows = []
    for label, vals in buckets.items():
        if not vals:
            continue
        rows.append(
            {
                "confidence_bin": label,
                "n": len(vals),
                "empirical_accuracy": sum(vals) / len(vals),
            }
        )
    return rows


def bridge_theory_overlap(
    preds: list[str],
    labels: list[str],
    *,
    min_count: int = 2,
) -> list[dict[str, Any]]:
    """Pairwise confusion frequencies for bridge-theory analysis (Fig. 4)."""
    pairs: Counter[tuple[str, str]] = Counter()
    for p, t in zip(preds, labels):
        if p != t:
            a, b = sorted([p, t])
            pairs[(a, b)] += 1
    rows = [
        {"theory_a": a, "theory_b": b, "confusion_count": n}
        for (a, b), n in pairs.most_common()
        if n >= min_count
    ]
    return rows


def simplex_entropy_map(cases: list[EthicalCase]) -> dict[str, Any]:
    """Normative simplex entropy per case (Fig. 3)."""
    entropies = []
    for c in cases:
        scores = infer_normative_scores(c)
        entropies.append(normative_entropy(scores))
    arr = np.array(entropies)
    return {
        "mean_entropy": float(arr.mean()),
        "min_entropy": float(arr.min()),
        "max_entropy": float(arr.max()),
        "high_ambiguity_fraction": float((arr > 0.9).mean()),
    }


def pluralism_report(
    model: StackedEnsembleModel,
    cases: list[EthicalCase],
    cfg: EthicalPluralismConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or EthicalPluralismConfig()
    metrics = evaluate_classifier(model, cases, cfg)
    raw_ent: list[float] = []
    scaled_ent: list[float] = []
    X = build_feature_matrix(
        cases,
        cfg,
        use_normative=model.feature_flags.get("normative", True),
        use_context=model.feature_flags.get("context", True),
        use_embeddings=model.feature_flags.get("embeddings", True),
    )
    for x in X:
        p = model.predict_proba_row(x)
        raw_ent.append(classification_entropy(p))
        p2 = temperature_scale_probs(p, cfg.temperature_scaling)
        scaled_ent.append(classification_entropy(p2))
    bridges = bridge_theory_overlap(metrics["predictions"], metrics["labels"])
    return {
        "classification": {
            "em_accuracy": metrics["em_accuracy"],
            "macro_f1": metrics["macro_f1"],
        },
        "entropy": {
            "raw_mean": float(np.mean(raw_ent)),
            "scaled_mean_T": float(np.mean(scaled_ent)),
            "temperature": cfg.temperature_scaling,
            "paper_scaled_mean": 0.61,
            "paper_raw_mean": 1.66,
        },
        "simplex_prior": simplex_entropy_map(cases),
        "confidence_stratification": confidence_stratification(model, cases, cfg),
        "bridge_theories": bridges[:12],
        "subtheory_schools": {cid: SUBTHEORY_BY_ID[cid].school for cid in SUBTHEORY_BY_ID},
    }
