"""Confidence-weighted Ridge reconstruction (paper §3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mhs_align.layout import MHS_ATTRIBUTES


def token_confidence(log_probs: dict[int, float], chosen: int) -> float:
    """Softmax confidence over ordinal label tokens."""
    keys = list(log_probs.keys())
    vals = np.array([log_probs[k] for k in keys], dtype=np.float64)
    vals = vals - vals.max()
    exp = np.exp(vals)
    probs = exp / exp.sum()
    idx = keys.index(chosen)
    return float(probs[idx])


def confidence_weighted_features(
    scores: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    """x_{n,i} = S_{n,i} * C_{n,i} per attribute."""
    return scores * confidences


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Closed-form Ridge weights (no intercept)."""
    d = x_train.shape[1]
    xtx = x_train.T @ x_train + alpha * np.eye(d)
    return np.linalg.solve(xtx, x_train.T @ y_train)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def synthetic_toy_split(
    n: int = 400,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic comment-level data mirroring paper structure:
    behavioral dims track hate; evaluative dims inverted in LLM channel.
    """
    rng = np.random.default_rng(seed)
    true_hate = rng.normal(0, 1.5, size=n)

    human_scores = np.zeros((n, len(MHS_ATTRIBUTES)))
    behavioral_idx = [MHS_ATTRIBUTES.index(a) for a in (
        "insult", "humiliate", "violence", "attack_defend", "dehumanize", "genocide"
    )]
    eval_idx = [MHS_ATTRIBUTES.index(a) for a in ("respect", "sentiment", "status", "hatespeech")]

    for j in behavioral_idx:
        human_scores[:, j] = np.clip(true_hate + rng.normal(0, 0.4, n), 0, 4)
    for j in eval_idx:
        human_scores[:, j] = np.clip(2 - true_hate + rng.normal(0, 0.4, n), 0, 4)

    llm_scores = human_scores.copy()
    for j in eval_idx:
        llm_scores[:, j] = np.clip(4 - human_scores[:, j] + rng.normal(0, 0.3, n), 0, 4)

    conf = rng.uniform(0.55, 0.95, size=(n, len(MHS_ATTRIBUTES)))
    conf[:, eval_idx] = rng.uniform(0.88, 0.95, size=(n, len(eval_idx)))

    y = true_hate + rng.normal(0, 0.15, n)
    return llm_scores, conf, y, human_scores


def ridge_reconstruction_smoke(alpha: float = 1.0) -> dict[str, Any]:
    llm_s, conf, y, _human = synthetic_toy_split()
    n = len(y)
    split = int(0.8 * n)
    x = confidence_weighted_features(llm_s, conf)
    w = fit_ridge(x[:split], y[:split], alpha=alpha)
    pred = x[split:] @ w
    r2 = r2_score(y[split:], pred)
    direct = r2_score(y[split:], llm_s[split:, MHS_ATTRIBUTES.index("hatespeech")])
    return {
        "n_train": split,
        "n_test": n - split,
        "r2_confidence_weighted_ridge": round(r2, 4),
        "r2_direct_hatespeech_attribute": round(direct, 4),
        "ridge_beats_direct": r2 > direct,
        "n_attributes": len(MHS_ATTRIBUTES),
    }
