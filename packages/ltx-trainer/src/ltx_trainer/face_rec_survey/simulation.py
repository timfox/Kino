"""Toy PCA vs nearest-neighbor recognition under synthetic confounders."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.face_rec_survey.challenges import ChallengeFactor
from ltx_trainer.face_rec_survey.config import FaceRecSurveyConfig


def _synthetic_gallery(
    cfg: FaceRecSurveyConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dim = cfg.image_size * cfg.image_size // 4
    n = cfg.n_subjects * cfg.n_images_per_subject
    labels = np.repeat(np.arange(cfg.n_subjects), cfg.n_images_per_subject)
    basis = rng.standard_normal((cfg.n_subjects, dim))
    noise = 0.15 * rng.standard_normal((n, dim))
    rows = np.arange(n) // cfg.n_images_per_subject
    x = basis[rows] + noise
    return x.astype(np.float64), labels


def _apply_confounder(x: np.ndarray, factor: ChallengeFactor, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    scale = {
        ChallengeFactor.ILLUMINATION: 0.35,
        ChallengeFactor.POSE: 0.28,
        ChallengeFactor.EXPRESSION: 0.22,
        ChallengeFactor.OCCLUSION: 0.40,
        ChallengeFactor.AGING: 0.30,
    }[factor]
    if factor == ChallengeFactor.ILLUMINATION:
        out *= rng.uniform(0.5, 1.5, size=(x.shape[0], 1))
    elif factor == ChallengeFactor.OCCLUSION:
        mask = rng.random(x.shape) > 0.25
        out = out * mask
    else:
        out += scale * rng.standard_normal(out.shape)
    return out


def _pca_nn_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_components: int,
) -> float:
    mu = train_x.mean(axis=0, keepdims=True)
    train_c = train_x - mu
    test_c = test_x - mu
    _, _, vt = np.linalg.svd(train_c, full_matrices=False)
    k = min(n_components, vt.shape[0])
    w = vt[:k].T
    train_p = train_c @ w
    test_p = test_c @ w
    correct = 0
    for i in range(test_p.shape[0]):
        dists = np.linalg.norm(train_p - test_p[i], axis=1)
        pred = train_y[int(np.argmin(dists))]
        if pred == test_y[i]:
            correct += 1
    return correct / max(test_p.shape[0], 1)


def confounder_accuracy_report(
    cfg: FaceRecSurveyConfig | None = None,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = cfg or FaceRecSurveyConfig()
    rng = np.random.default_rng(seed)
    x, y = _synthetic_gallery(cfg, rng)
    n = x.shape[0]
    idx = rng.permutation(n)
    split = int(0.7 * n)
    tr, te = idx[:split], idx[split:]
    baseline = _pca_nn_accuracy(x[tr], y[tr], x[te], y[te], cfg.n_pca_components)
    by_factor: dict[str, float] = {}
    for factor in ChallengeFactor:
        x_te = _apply_confounder(x[te], factor, rng)
        by_factor[factor.value] = round(
            _pca_nn_accuracy(x[tr], y[tr], x_te, y[te], cfg.n_pca_components),
            4,
        )
    return {
        "baseline_accuracy": round(float(baseline), 4),
        "confounded_accuracy": by_factor,
        "largest_drop_factor": min(by_factor, key=by_factor.get),  # type: ignore[arg-type]
        "n_train": int(split),
        "n_test": int(n - split),
    }
