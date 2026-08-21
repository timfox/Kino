"""Phoneme embedding-space analysis (§4.4, Table 3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ssl_soft_infer.config import SslSoftInferConfig
from ltx_trainer.ssl_soft_infer.posterior import token_embedding


def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def intra_class_variance(vectors: np.ndarray) -> float:
    """Eq. (6) on L2-normalized frame embeddings."""
    z = l2_normalize(np.asarray(vectors, dtype=np.float64))
    mu = np.mean(z, axis=0)
    mu = mu / np.linalg.norm(mu)
    return float(np.mean(np.sum((z - mu) ** 2, axis=1)))


def inter_class_distance(mu_p: np.ndarray, mu_q: np.ndarray) -> float:
    """Eq. (7) between normalized class means."""
    a = mu_p / np.linalg.norm(mu_p)
    b = mu_q / np.linalg.norm(mu_q)
    return float(np.sum((a - b) ** 2))


def fisher_ratio(intra: float, inter: float) -> float:
    if intra <= 0:
        return float("inf")
    return inter / intra


def phoneme_separability_demo(seed: int = 0, cfg: SslSoftInferConfig | None = None) -> dict[str, Any]:
    """Synthetic 3-phoneme toy separability under hard vs soft assignment."""
    c = cfg or SslSoftInferConfig()
    rng = np.random.default_rng(seed)
    dim = 16
    k = 6
    # Three well-separated prototype regions (two centroids each).
    prototypes = np.stack(
        [
            np.pad(np.array([1.0, 0.2, 0.0]), (0, dim - 3)),
            np.pad(np.array([0.9, 0.3, 0.1]), (0, dim - 3)),
            np.pad(np.array([0.0, 1.0, 0.1]), (0, dim - 3)),
            np.pad(np.array([0.1, 0.9, 0.2]), (0, dim - 3)),
            np.pad(np.array([0.0, 0.1, 1.0]), (0, dim - 3)),
            np.pad(np.array([0.1, 0.0, 0.9]), (0, dim - 3)),
        ]
    )
    centroids = prototypes + rng.normal(scale=0.02, size=prototypes.shape)
    embeddings = centroids.copy()

    phonemes = ("AA", "IY", "SH")
    phoneme_centroid_idx = {"AA": (0, 1), "IY": (2, 3), "SH": (4, 5)}
    hard_intras: list[float] = []
    soft_intras: list[float] = []
    class_means_hard: dict[str, np.ndarray] = {}
    class_means_soft: dict[str, np.ndarray] = {}

    for ph in phonemes:
        idx_a, idx_b = phoneme_centroid_idx[ph]
        frames = np.vstack(
            [
                centroids[idx_a] + rng.normal(scale=0.08, size=dim),
                centroids[idx_b] + rng.normal(scale=0.08, size=dim),
            ]
            * 6
        )
        zh = np.stack(
            [token_embedding(f, centroids, embeddings, mode="hard") for f in frames],
            axis=0,
        )
        zs = np.stack(
            [
                token_embedding(f, centroids, embeddings, mode="soft", tau=c.tau_librispeech)
                for f in frames
            ],
            axis=0,
        )
        hard_intras.append(intra_class_variance(zh))
        soft_intras.append(intra_class_variance(zs))
        class_means_hard[ph] = l2_normalize(np.mean(l2_normalize(zh), axis=0))
        class_means_soft[ph] = l2_normalize(np.mean(l2_normalize(zs), axis=0))

    hard_inters = [
        inter_class_distance(class_means_hard[a], class_means_hard[b])
        for i, a in enumerate(phonemes)
        for b in phonemes[i + 1 :]
    ]
    soft_inters = [
        inter_class_distance(class_means_soft[a], class_means_soft[b])
        for i, a in enumerate(phonemes)
        for b in phonemes[i + 1 :]
    ]

    hard_intra = float(np.mean(hard_intras))
    soft_intra = float(np.mean(soft_intras))
    hard_inter = float(np.mean(hard_inters))
    soft_inter = float(np.mean(soft_inters))
    hard_ratio = fisher_ratio(hard_intra, hard_inter)
    soft_ratio = fisher_ratio(soft_intra, soft_inter)

    return {
        "hard": {"intra": hard_intra, "inter": hard_inter, "ratio": hard_ratio},
        "soft": {"intra": soft_intra, "inter": soft_inter, "ratio": soft_ratio},
        "ratio_improved": soft_ratio > hard_ratio,
    }
