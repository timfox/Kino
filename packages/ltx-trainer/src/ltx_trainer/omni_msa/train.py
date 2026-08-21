"""Stub discriminative readout training on synthetic MOSI batch (arXiv:2606.05713)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.omni_msa.metrics import mae, normalize_labels, pearson_corr, regression_bundle
from ltx_trainer.omni_msa.mock import synthetic_mosi_batch
from ltx_trainer.omni_msa.readout import DiscriminativeHead, RegressionHeadConfig, pool_last_non_pad


def _extract_features(head: DiscriminativeHead, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = pool_last_non_pad(hidden, mask)
    x = np.asarray(z, dtype=np.float64).reshape(-1)
    h = head.w1 @ x + head.b1
    h = head._layer_norm(h)
    return np.maximum(h, 0.0)


def fit_readout_head(
    head: DiscriminativeHead,
    hidden_states: np.ndarray,
    attention_masks: np.ndarray,
    labels_norm: np.ndarray,
) -> dict[str, float]:
    """Closed-form fit of final linear layer on fixed random MLP features (Eq. 3 stub)."""
    hs = np.asarray(hidden_states)
    masks = np.asarray(attention_masks)
    y = np.asarray(labels_norm, dtype=np.float64).reshape(-1)
    features = np.stack([_extract_features(head, hs[i], masks[i]) for i in range(hs.shape[0])])
    design = np.hstack([features, np.ones((features.shape[0], 1))])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    head.w2 = coef[:-1].reshape(1, -1)
    head.b2 = np.array([coef[-1]])
    preds_norm = (features @ head.w2.T + head.b2).reshape(-1)
    return {
        "train_mae_norm": round(float(np.mean(np.abs(preds_norm - y))), 4),
        "train_corr_norm": round(pearson_corr(y, preds_norm), 4),
    }


def train_discriminative_stub(
    *,
    seed: int = 0,
    n: int = 128,
    mu: float | None = None,
    sigma: float | None = None,
) -> dict[str, Any]:
    """One-epoch stub: fit MLP readout head on synthetic pooled hidden states."""
    batch = synthetic_mosi_batch(seed=seed, n=n)
    mu_v = batch["mu"] if mu is None else mu
    sigma_v = batch["sigma"] if sigma is None else sigma
    labels_norm = normalize_labels(batch["labels"], mu_v, sigma_v)

    d = int(batch["hidden_states"].shape[-1])
    head = DiscriminativeHead(RegressionHeadConfig(hidden_size=d, head_hidden=min(32, d)), seed=seed)

    preds_before = head.predict_batch(batch["hidden_states"], batch["attention_masks"], mu=mu_v, sigma=sigma_v)
    metrics_before = regression_bundle(batch["labels"], preds_before)

    fit_stats = fit_readout_head(head, batch["hidden_states"], batch["attention_masks"], labels_norm)
    preds_after = head.predict_batch(batch["hidden_states"], batch["attention_masks"], mu=mu_v, sigma=sigma_v)
    metrics_after = regression_bundle(batch["labels"], preds_after)

    return {
        "n_samples": n,
        "before": metrics_before,
        "after": metrics_after,
        "fit": fit_stats,
        "mae_improved": metrics_after["mae"] <= metrics_before["mae"],
        "loss": "L1 on normalized labels (Eq. 6 stub)",
    }
