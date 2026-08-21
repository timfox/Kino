"""Proxy-to-wild domain gap analysis stub (§2, Fig. 2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dsfa.config import DsfaConfig


def distribution_overlap(a: np.ndarray, b: np.ndarray, *, bins: int = 50) -> float:
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.sum(np.minimum(ha, hb)) * 100.0)


def proxy_wild_gap_demo(*, seed: int = 0, cfg: DsfaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DsfaConfig()
    rng = np.random.default_rng(seed)
    cors_mu = rng.normal(0, 1, 1000)
    cosg_mu = cors_mu + rng.normal(0.3, 0.5, 1000)
    dsfa_mu = cors_mu + rng.normal(0.15, 0.35, 1000)

    cors_std = rng.normal(1, 0.2, 1000)
    cosg_std = cors_std + rng.normal(0.25, 0.15, 1000)
    dsfa_std = cors_std + rng.normal(0.12, 0.12, 1000)

    baseline_mu_overlap = distribution_overlap(cors_mu, cosg_mu)
    dsfa_mu_overlap = distribution_overlap(dsfa_mu, cosg_mu)
    baseline_std_overlap = distribution_overlap(cors_std, cosg_std)
    dsfa_std_overlap = distribution_overlap(dsfa_std, cosg_std)

    return {
        "artifact_mismatch": True,
        "baseline_mean_overlap_pct": round(baseline_mu_overlap, 2),
        "dsfa_mean_overlap_pct": round(dsfa_mu_overlap, 2),
        "baseline_std_overlap_pct": round(baseline_std_overlap, 2),
        "dsfa_std_overlap_pct": round(dsfa_std_overlap, 2),
        "paper_mean_overlap_gain": [cfg.mean_overlap_baseline, cfg.mean_overlap_dsfa],
        "paper_std_overlap_gain": [cfg.std_overlap_baseline, cfg.std_overlap_dsfa],
    }
