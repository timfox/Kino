"""Toy smoke: FCP filter, G-loss, sync delay, pseudo-label."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ctrnet.config import CtrnetConfig
from ltx_trainer.ctrnet.fcp import estimate_fcp_filter
from ltx_trainer.ctrnet.losses import g_loss, overlap_sampling_weight, speaker_activity_loss
from ltx_trainer.ctrnet.pulss import pseudo_label_at_reference


def evaluation_smoke(cfg: CtrnetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CtrnetConfig()
    rng = np.random.default_rng(0)
    n = 64
    clean = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    cross = 0.35 * np.roll(clean, 2)
    mixture = clean + cross + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    h = estimate_fcp_filter(
        mixture,
        clean,
        past_taps=cfg.fcp_past_taps_i,
        future_taps=cfg.fcp_future_taps_j,
        xi=cfg.fcp_xi,
        use_quantile_90=True,
    )
    recon = np.zeros(n, dtype=np.complex128)
    for t in range(cfg.fcp_past_taps_i, n - cfg.fcp_future_taps_j):
        stack = [clean[t + lag] for lag in range(-cfg.fcp_past_taps_i, cfg.fcp_future_taps_j + 1)]
        recon[t] = np.vdot(h, np.asarray(stack, dtype=np.complex128))

    pl, k_hat = pseudo_label_at_reference(
        clean,
        mixture,
        max_delay=cfg.max_sync_delay_frames_e,
        filter_taps=cfg.pseudo_label_filter_taps_l,
    )

    activity = np.ones(n, dtype=np.float64)
    activity[:8] = 0.0
    sa = speaker_activity_loss(
        clean * (1.0 - activity),
        activity,
        alpha=cfg.magnitude_compress_alpha,
        mixture_ref=mixture,
    )

    active_counts = np.array([1, 1, 2, 3, 4, 2, 1, 1] * (n // 8), dtype=np.float64)
    w = overlap_sampling_weight(active_counts, theta=float(cfg.overlap_sampling_theta))

    return {
        "fcp_filter_taps": int(h.size),
        "g_loss_recon": round(g_loss(mixture, recon, alpha=cfg.magnitude_compress_alpha), 4),
        "sa_loss_inactive": round(sa, 4),
        "sync_delay_frames": k_hat,
        "pseudo_label_g_loss": round(
            g_loss(mixture, pl, alpha=cfg.magnitude_compress_alpha, normalize_den=mixture),
            4,
        ),
        "overlap_weight_high_overlap": round(w, 2),
        "paper_pulss_cpwer_test_oracle_pct": 19.5,
        "paper_gss_cpwer_test_oracle_pct": 29.7,
    }
