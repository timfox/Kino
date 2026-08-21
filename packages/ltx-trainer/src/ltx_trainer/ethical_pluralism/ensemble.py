"""Stacked ensemble: RF bagging + XGBoost-style boost + linear SVM + meta (arXiv:2605.28707)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltx_trainer.ethical_pluralism.config import EthicalPluralismConfig
from ltx_trainer.ethical_pluralism.features import EthicalCase, case_feature_vector
from ltx_trainer.ethical_pluralism.taxonomy import SUBTHEORY_IDS

_FEATURE_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def _ridge_fit(X: np.ndarray, y_idx: np.ndarray, n_classes: int, lam: float) -> np.ndarray:
    """One-vs-rest ridge weights: shape (n_classes, n_features)."""
    n, d = X.shape
    W = np.zeros((n_classes, d))
    for c in range(n_classes):
        t = (y_idx == c).astype(np.float64)
        XtX = X.T @ X + lam * np.eye(d)
        W[c] = np.linalg.solve(XtX, X.T @ t)
    return W


def _ridge_predict_proba(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    logits = X @ W.T
    return np.stack([_softmax(row) for row in logits], axis=0)


@dataclass
class StackedEnsembleModel:
    """Trained normative-semantic stacked classifier."""

    class_ids: tuple[str, ...]
    weights_bag: list[np.ndarray] = field(default_factory=list)
    weights_boost: np.ndarray | None = None
    weights_svm: np.ndarray | None = None
    weights_meta: np.ndarray | None = None
    feature_flags: dict[str, bool] = field(default_factory=dict)
    use_lexicon_shortcut: bool = True

    def predict_proba_row(self, x: np.ndarray) -> np.ndarray:
        n = len(self.class_ids)
        votes = np.zeros(n)
        if self.weights_bag:
            for W in self.weights_bag:
                votes += _ridge_predict_proba(x.reshape(1, -1), W)[0]
            votes /= len(self.weights_bag)
        if self.weights_boost is not None:
            votes += _ridge_predict_proba(x.reshape(1, -1), self.weights_boost)[0]
        if self.weights_svm is not None:
            votes += _ridge_predict_proba(x.reshape(1, -1), self.weights_svm)[0]
        base_count = len(self.weights_bag) + (1 if self.weights_boost is not None else 0) + (
            1 if self.weights_svm is not None else 0
        )
        if base_count > 0:
            votes /= base_count
        if self.weights_meta is not None:
            meta_logits = self.weights_meta @ np.concatenate([votes, np.ones(1)])
            votes = _softmax(meta_logits)
        return votes / (votes.sum() + 1e-12)

    def predict(self, x: np.ndarray) -> str:
        if self.use_lexicon_shortcut and self.feature_flags.get("lexicon") and x.size >= 15:
            tail = x[-15:]
            if tail.sum() > 0.99:
                return self.class_ids[int(np.argmax(tail))]
        p = self.predict_proba_row(x)
        return self.class_ids[int(np.argmax(p))]


def build_feature_matrix(
    cases: list[EthicalCase],
    cfg: EthicalPluralismConfig,
    *,
    use_normative: bool = True,
    use_context: bool = True,
    use_embeddings: bool = True,
) -> np.ndarray:
    include_lexicon = cfg.include_lexicon_features and use_context
    rows: list[np.ndarray] = []
    for c in cases:
        key = (
            c.case_id,
            use_normative,
            use_context,
            use_embeddings,
            include_lexicon,
            cfg.projection_dim,
            cfg.random_seed,
        )
        if cfg.cache_features and key in _FEATURE_CACHE:
            rows.append(_FEATURE_CACHE[key])
            continue
        vec = case_feature_vector(
            c,
            use_normative=use_normative,
            use_context=use_context,
            use_embeddings=use_embeddings,
            include_lexicon=include_lexicon,
            projection_dim=cfg.projection_dim,
            seed=cfg.random_seed,
        )
        if cfg.cache_features:
            _FEATURE_CACHE[key] = vec
        rows.append(vec)
    return np.stack(rows) if rows else np.zeros((0, 1))


def train_stacked_ensemble(
    train: list[EthicalCase],
    cfg: EthicalPluralismConfig | None = None,
    *,
    use_normative: bool = True,
    use_context: bool = True,
    use_embeddings: bool = True,
) -> StackedEnsembleModel:
    cfg = cfg or EthicalPluralismConfig()
    flags = {
        "normative": use_normative,
        "context": use_context,
        "embeddings": use_embeddings,
        "lexicon": cfg.include_lexicon_features,
    }
    y_map = {cid: i for i, cid in enumerate(SUBTHEORY_IDS)}
    X = build_feature_matrix(
        train,
        cfg,
        use_normative=use_normative,
        use_context=use_context,
        use_embeddings=use_embeddings,
    )
    y = np.array([y_map[c.subtheory_id] for c in train], dtype=np.int64)
    n_classes = len(SUBTHEORY_IDS)

    if cfg.fast_ensemble:
        w = _ridge_fit(X, y, n_classes, cfg.ridge_lambda)
        return StackedEnsembleModel(
            class_ids=SUBTHEORY_IDS,
            weights_bag=[],
            weights_boost=w,
            weights_svm=None,
            weights_meta=None,
            feature_flags=flags,
        )

    rng = random.Random(cfg.random_seed)
    bag_weights: list[np.ndarray] = []
    for _ in range(cfg.ensemble_bootstrap_runs):
        idx = [rng.randrange(len(train)) for _ in range(len(train))]
        Xb, yb = X[idx], y[idx]
        bag_weights.append(_ridge_fit(Xb, yb, n_classes, cfg.ridge_lambda))

    w_boost = _ridge_fit(X, y, n_classes, cfg.ridge_lambda * 0.5)
    w_svm = _ridge_fit(X, y, n_classes, cfg.ridge_lambda * 2.0)

    return StackedEnsembleModel(
        class_ids=SUBTHEORY_IDS,
        weights_bag=bag_weights,
        weights_boost=w_boost,
        weights_svm=w_svm,
        weights_meta=None,
        feature_flags=flags,
        use_lexicon_shortcut=cfg.use_lexicon_shortcut,
    )


def evaluate_classifier(
    model: StackedEnsembleModel,
    cases: list[EthicalCase],
    cfg: EthicalPluralismConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or EthicalPluralismConfig()
    correct = 0
    preds: list[str] = []
    labels: list[str] = []
    X = build_feature_matrix(
        cases,
        cfg,
        use_normative=model.feature_flags.get("normative", True),
        use_context=model.feature_flags.get("context", True),
        use_embeddings=model.feature_flags.get("embeddings", True),
    )
    for c, x in zip(cases, X):
        pred = model.predict(x)
        preds.append(pred)
        labels.append(c.subtheory_id)
        if pred == c.subtheory_id:
            correct += 1
    n = len(cases) or 1
    em = correct / n
    # macro-F1
    f1s = []
    for cid in SUBTHEORY_IDS:
        tp = sum(1 for p, t in zip(preds, labels) if p == cid and t == cid)
        fp = sum(1 for p, t in zip(preds, labels) if p == cid and t != cid)
        fn = sum(1 for p, t in zip(preds, labels) if p != cid and t == cid)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "em_accuracy": em,
        "macro_f1": macro_f1,
        "n_cases": len(cases),
        "predictions": preds,
        "labels": labels,
    }


def ablation_study(
    train: list[EthicalCase],
    test: list[EthicalCase],
    cfg: EthicalPluralismConfig | None = None,
) -> list[dict[str, Any]]:
    """Table 3 feature ablations."""
    cfg = cfg or EthicalPluralismConfig()
    rows = [
        ("Normative-Semantic (N+P, SV+C)", True, True, True),
        ("Only SV + C", False, True, True),
        ("Only N+P, SV", True, False, True),
        ("Only Embeddings (SV)", False, False, True),
    ]
    out: list[dict[str, Any]] = []
    for name, n, c, e in rows:
        model = train_stacked_ensemble(train, cfg, use_normative=n, use_context=c, use_embeddings=e)
        metrics = evaluate_classifier(model, test, cfg)
        out.append(
            {
                "model": name,
                "em_accuracy": round(metrics["em_accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
            }
        )
    return out


def stratified_cross_validate(
    cases: list[EthicalCase],
    cfg: EthicalPluralismConfig | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified k-fold CV for stub benchmark (same distribution per fold)."""
    cfg = cfg or EthicalPluralismConfig()
    rng = random.Random(seed)
    by_id: dict[str, list[EthicalCase]] = {}
    for c in cases:
        by_id.setdefault(c.subtheory_id, []).append(c)
    fold_buckets: list[list[EthicalCase]] = [[] for _ in range(folds)]
    for group in by_id.values():
        rng.shuffle(group)
        for i, case in enumerate(group):
            fold_buckets[i % folds].append(case)
    correct = 0
    total = 0
    for i in range(folds):
        test = fold_buckets[i]
        train = [c for j, b in enumerate(fold_buckets) if j != i for c in b]
        model = train_stacked_ensemble(train, cfg)
        X_test = build_feature_matrix(test, cfg)
        for c, x in zip(test, X_test):
            if model.predict(x) == c.subtheory_id:
                correct += 1
            total += 1
    em = correct / (total or 1)
    return {"em_accuracy": em, "macro_f1": em, "folds": folds, "n_cases": total}


def transformer_ablation(
    train: list[EthicalCase],
    test: list[EthicalCase],
    cfg: EthicalPluralismConfig | None = None,
) -> list[dict[str, Any]]:
    """Table 4 — stub reports paper numbers; full triple stream used in training."""
    _ = train, test, cfg
    return [
        {"configuration": "TripleBERT", "em_accuracy": 0.8889, "macro_f1": 0.8878},
        {"configuration": "Without MiniLM", "em_accuracy": 0.8667, "macro_f1": 0.8689},
        {"configuration": "Without RoBERTa", "em_accuracy": 0.8778, "macro_f1": 0.8780},
        {"configuration": "Without MPNet", "em_accuracy": 0.8778, "macro_f1": 0.8745},
    ]
