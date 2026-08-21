"""Structural analysis metrics stub (§7, Table 2)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ltx_trainer.eisbach.config import EisbachConfig


def self_similarity_matrix(mel: np.ndarray) -> np.ndarray:
    """Cosine similarity between mel frames."""
    mel_n = mel / (np.linalg.norm(mel, axis=1, keepdims=True) + 1e-8)
    return mel_n @ mel_n.T


def pca_coverage(mel: np.ndarray) -> float:
    """Area proxy: std of first two PCA components."""
    mel_c = mel - mel.mean(axis=0)
    _, _, vt = np.linalg.svd(mel_c, full_matrices=False)
    proj = mel_c @ vt[:2].T
    return float(np.std(proj[:, 0]) * np.std(proj[:, 1]))


def dynamic_range_db(mel: np.ndarray) -> float:
    peak = float(np.max(mel))
    floor = float(np.percentile(mel, 5))
    return 20.0 * np.log10(max(peak, 1e-8) / max(floor, 1e-8))


def structural_metrics(
    mel: np.ndarray,
    *,
    model: Literal["barrier", "baseline"],
    cfg: EisbachConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or EisbachConfig()
    sim = self_similarity_matrix(mel)
    off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
    return {
        "self_sim_off_diag_mean": float(np.mean(off_diag)),
        "pca_coverage": pca_coverage(mel),
        "dynamic_range_db": dynamic_range_db(mel),
        "model": 1.0 if model == "barrier" else 0.0,
        "expected_dynamic_range_db": (
            cfg.barrier_dynamic_range_db if model == "barrier" else cfg.baseline_dynamic_range_db
        ),
    }


def analysis_demo(*, seed: int = 0, cfg: EisbachConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    rng = np.random.default_rng(seed)
    t, f = 600, 128
    barrier_mel = rng.normal(size=(t, f))
    for seg in [(0, 120), (180, 300), (360, 480)]:
        barrier_mel[seg[0] : seg[1], :] *= rng.uniform(1.5, 3.0)
    baseline_mel = rng.normal(size=(t, f)) * 0.5 + 0.2
    b = structural_metrics(barrier_mel, model="barrier", cfg=cfg)
    base = structural_metrics(baseline_mel, model="baseline", cfg=cfg)
    return {
        "barrier_pca_coverage": b["pca_coverage"],
        "baseline_pca_coverage": base["pca_coverage"],
        "barrier_more_development": b["pca_coverage"] > base["pca_coverage"],
    }
