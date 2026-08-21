"""Toy omni-modal MSA smoke and evaluation_smoke (arXiv:2606.05713)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.omni_msa.metrics import knn_label_smoothness, mae, pearson_corr, regression_bundle
from ltx_trainer.omni_msa.readout import (
    DiscriminativeHead,
    ReadoutMode,
    RegressionHeadConfig,
    generative_decode,
    generative_predict,
    last_non_pad_index,
    pool_last_non_pad,
)


def synthetic_mosi_batch(*, seed: int = 0, n: int = 64) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    labels = rng.uniform(-2.5, 2.5, size=n)
    d = 32
    # sentiment-correlated embeddings for kNN smoke
    embeddings = rng.standard_normal((n, d))
    embeddings += (labels / 3.0)[:, None] * 0.8
    seq_len = 12
    hidden = rng.standard_normal((n, seq_len, d))
    hidden += (labels / 3.0)[:, None, None] * 0.5
    masks = np.ones((n, seq_len), dtype=np.int64)
    masks[:, -2:] = 0  # right padding
    return {
        "labels": labels,
        "embeddings": embeddings,
        "hidden_states": hidden,
        "attention_masks": masks,
        "mu": float(np.mean(labels)),
        "sigma": float(np.std(labels) + 1e-6),
    }


def run_discriminative_smoke(*, seed: int = 0) -> dict[str, Any]:
    batch = synthetic_mosi_batch(seed=seed, n=48)
    head = DiscriminativeHead(
        RegressionHeadConfig(hidden_size=batch["hidden_states"].shape[-1], head_hidden=16),
        seed=seed,
    )
    preds = head.predict_batch(
        batch["hidden_states"],
        batch["attention_masks"],
        mu=batch["mu"],
        sigma=batch["sigma"],
    )
    metrics = regression_bundle(batch["labels"], preds)
    knn = knn_label_smoothness(batch["embeddings"], batch["labels"], k=10, seed=seed)
    sample_mask = batch["attention_masks"][0]
    z = pool_last_non_pad(batch["hidden_states"][0], sample_mask)
    return {
        "last_token_index": last_non_pad_index(sample_mask),
        "pooled_dim": int(z.shape[0]),
        "metrics": metrics,
        "knn_smoothness": knn,
    }


def run_generative_smoke(*, seed: int = 0, n: int = 100) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, 5))
    zs = [generative_decode(logits[i], mode=ReadoutMode.GENERATIVE_ZERO_SHOT, seed=seed + i) for i in range(n)]
    ts = [generative_decode(logits[i], mode=ReadoutMode.GENERATIVE_TRAINED, seed=seed + i + 1000) for i in range(n)]
    _, rel_z = generative_predict(zs)
    _, rel_t = generative_predict(ts)
    return {
        "zero_shot_unparsable_pct": rel_z["unparsable_pct"],
        "zero_shot_oob_pct": rel_z["oob_pct"],
        "trained_unparsable_pct": rel_t["unparsable_pct"],
        "sample_zero_shot": zs[:3],
    }


def compare_readouts_smoke(*, seed: int = 0) -> dict[str, Any]:
    batch = synthetic_mosi_batch(seed=seed, n=32)
    head = DiscriminativeHead(
        RegressionHeadConfig(hidden_size=batch["hidden_states"].shape[-1], head_hidden=16),
        seed=seed,
    )
    disc_preds = head.predict_batch(batch["hidden_states"], batch["attention_masks"], mu=batch["mu"], sigma=batch["sigma"])
    gen_strings = [
        generative_decode(np.zeros(3), mode=ReadoutMode.GENERATIVE_ZERO_SHOT, seed=seed + i)
        for i in range(len(batch["labels"]))
    ]
    gen_preds, rel = generative_predict(gen_strings)
    return {
        "disc_mae": round(mae(batch["labels"], disc_preds), 3),
        "gen_mae": round(mae(batch["labels"], gen_preds), 3),
        "disc_corr": round(pearson_corr(batch["labels"], disc_preds), 3),
        "gen_corr": round(pearson_corr(batch["labels"], gen_preds), 3),
        "gen_unparsable_pct": rel["unparsable_pct"],
        "disc_unparsable_pct": 0.0,
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    disc = run_discriminative_smoke(seed=seed)
    gen = run_generative_smoke(seed=seed + 1)
    cmp = compare_readouts_smoke(seed=seed + 2)
    return {
        "discriminative": disc,
        "generative": gen,
        "comparison": cmp,
        "disc_mae_lt_gen_mae": cmp["disc_mae"] < cmp["gen_mae"],
    }
