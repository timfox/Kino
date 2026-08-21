"""Hidden-number linear probe and vision/language gap metrics (Eq. 1, §2.3)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.vlm_count.regimes import regime_for_n


def hidden_number_from_probe_preds(preds: np.ndarray) -> int:
    """Aggregate per-patch stone presence predictions into NH."""
    return int(np.sum(preds > 0.5))


def vision_gap(n_hidden: int, n_ground: int) -> float:
    return float(abs(n_hidden - n_ground))


def language_gap(n_hidden: int, n_pred: int) -> float:
    return float(abs(n_hidden - n_pred))


def linear_probe_predict(
    embeddings: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    """Binary probe: stone present at patch."""
    logits = embeddings @ weights + bias
    return (logits > 0).astype(np.float64)


def fit_id_probe(
    embeddings_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Train pooled linear probe on ID patches only (stub: least squares on stacked patches)."""
    rng = rng or np.random.default_rng(0)
    xs, ys = [], []
    for emb, lab in zip(embeddings_list, labels_list, strict=True):
        xs.append(emb.reshape(-1, emb.shape[-1]))
        ys.append(lab.reshape(-1))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    d = x.shape[1]
    w = rng.standard_normal(d) * 0.01
    for _ in range(50):
        pred = x @ w
        grad = x.T @ (pred - y) / len(y) + 0.01 * w
        w -= 0.1 * grad
    b = float(np.median(y - x @ w))
    return w, b


def diagnose_sample(
    n_ground: int,
    n_hidden: int,
    n_pred: int,
    *,
    visual_train_max: int = 49,
) -> dict[str, float | str]:
    regime = regime_for_n(n_ground, visual_train_max=visual_train_max)
    vg = vision_gap(n_hidden, n_ground)
    lg = language_gap(n_hidden, n_pred)
    return {
        "regime": regime,
        "n_ground": n_ground,
        "n_hidden": n_hidden,
        "n_pred": n_pred,
        "vision_gap": vg,
        "language_gap": lg,
        "bottleneck_stage": "symbolic_mapping" if vg < 1.0 and lg > 5 else "perception_or_magnitude",
    }
