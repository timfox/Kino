"""Distribution matching: weighted kNN subsampling (§2.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mv2lrs3.config import Mv2Lrs3Config


def feature_vector(
    *,
    duration_s: float,
    age: float,
    gender: int,
    skin_tone: int,
    yaw_mean: float,
    yaw_std: float,
    snr_db: float,
    speech_rate: float,
    cfg: Mv2Lrs3Config | None = None,
) -> np.ndarray:
    cfg = cfg or Mv2Lrs3Config()
    raw = np.array(
        [duration_s, age, gender, skin_tone, yaw_mean, yaw_std, snr_db, speech_rate],
        dtype=np.float64,
    )
    weights = np.array([cfg.factor_weights[f] for f in cfg.matching_factors], dtype=np.float64)
    return raw * np.sqrt(weights)


def weighted_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def knn_match_one(
    reference: np.ndarray,
    pool: np.ndarray,
    *,
    k: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Pick 1 of k nearest pool vectors (stochastic MV2LRS3 selection)."""
    dists = np.array([weighted_euclidean(reference, p) for p in pool])
    nn_idx = np.argsort(dists)[:k]
    rng = np.random.default_rng(seed)
    chosen = int(rng.choice(nn_idx))
    return {
        "nearest_indices": nn_idx.tolist(),
        "chosen_index": chosen,
        "distance": float(dists[chosen]),
        "k": k,
    }


def matching_demo(*, seed: int = 0, pool_size: int = 200) -> dict[str, Any]:
    cfg = Mv2Lrs3Config()
    rng = np.random.default_rng(seed)
    ref = feature_vector(
        duration_s=4.2,
        age=45.0,
        gender=1,
        skin_tone=6,
        yaw_mean=12.0,
        yaw_std=8.0,
        snr_db=35.0,
        speech_rate=3.2,
        cfg=cfg,
    )
    pool = np.stack(
        [
            feature_vector(
                duration_s=float(rng.uniform(2, 7)),
                age=float(rng.uniform(20, 80)),
                gender=int(rng.integers(0, 2)),
                skin_tone=int(rng.integers(5, 10)),
                yaw_mean=float(rng.uniform(0, 45)),
                yaw_std=float(rng.uniform(2, 20)),
                snr_db=float(rng.uniform(10, 50)),
                speech_rate=float(rng.uniform(1.5, 5.0)),
                cfg=cfg,
            )
            for _ in range(pool_size)
        ]
    )
    match = knn_match_one(ref, pool, k=cfg.knn_candidates, seed=seed)
    return {"reference_dim": len(ref), "pool_size": pool_size, "match": match}
