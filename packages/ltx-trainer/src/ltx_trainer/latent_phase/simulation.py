"""Synthetic latent-phase diagnosis demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.latent_phase.anomaly import ad_improvement, ood_fpr95_results
from ltx_trainer.latent_phase.config import LatentPhaseConfig
from ltx_trainer.latent_phase.diagnostics import (
    block_spin_batch,
    cluster_stability_proxy,
    k_not_order_parameters,
    overlap_summary,
    overlap_susceptibility,
    sample_latent_codes,
    susceptibility_sweep,
)
from ltx_trainer.latent_phase.generation import generation_improvement
from ltx_trainer.latent_phase.hyperspherical import hyperspherical_demo, taylor_decay_ratio


def phase_diagnosis_demo(*, seed: int = 0, cfg: LatentPhaseConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentPhaseConfig()
    dim = cfg.prior.latent_dim
    rng = np.random.default_rng(seed)

    dis = sample_latent_codes(128, dim, phase="disordered", seed=seed)
    ord_codes = sample_latent_codes(128, dim, phase="ordered", seed=seed + 1)
    edge = sample_latent_codes(128, dim, phase="edge", seed=seed + 2)

    labels = rng.integers(0, 2, size=128)
    chi_sweep = susceptibility_sweep(alpha_star=cfg.edge_alpha_star)
    chi_vals = [r["chi_overlap"] for r in chi_sweep]
    alpha_star_idx = int(np.argmax(chi_vals))

    return {
        "overlap_disordered": overlap_summary(dis, seed=seed),
        "overlap_ordered": overlap_summary(ord_codes, seed=seed),
        "overlap_edge": overlap_summary(edge, seed=seed),
        "chi_sweep": chi_sweep,
        "alpha_star": float(chi_sweep[alpha_star_idx]["alpha"]),
        "chi_peak": float(max(chi_vals)),
        "block_spin_ordered_stability": cluster_stability_proxy(
            ord_codes, labels, n_blocks=cfg.prior.block_spin_dim
        ),
        "block_spin_disordered_stability": cluster_stability_proxy(
            dis, labels, n_blocks=cfg.prior.block_spin_dim
        ),
        "k_not_ordered": k_not_order_parameters(ord_codes),
        "k_not_disordered": k_not_order_parameters(dis),
        "block_spin_shape": block_spin_batch(ord_codes[:4], n_blocks=3).shape,
        "generation_cifar10": generation_improvement("CIFAR-10"),
        "ad_galaxy_zoo": ad_improvement("Galaxy Zoo 64"),
        "ood_fpr95": ood_fpr95_results(),
        "hyperspherical": hyperspherical_demo(cfg),
        "taylor_decay": taylor_decay_ratio(seed=seed),
    }
