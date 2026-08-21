"""Smoke evaluation for causal ST-SFR stub."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ltx_trainer.causal_st_sfr.config import CausalStSfrConfig
from ltx_trainer.causal_st_sfr.kernel import diffuse_far_field_normalized_coherence, kappa_band_limited
from ltx_trainer.causal_st_sfr.layout import fibonacci_sphere_points
from ltx_trainer.causal_st_sfr.lmmse import build_covariance_blocks, lmmse_posterior_mean


def evaluation_smoke(cfg: CausalStSfrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CausalStSfrConfig()
    w1 = 2.0 * math.pi * cfg.f1_hz
    w2 = 2.0 * math.pi * cfg.f2_hz
    k0 = float(kappa_band_limited(0.0, omega1=w1, omega2=w2))
    c00 = float(
        diffuse_far_field_normalized_coherence(0.0, omega=0.5 * (w1 + w2), c_sound=cfg.c_sound)
    )

    ts = 1.0 / cfg.fs_hz
    mics = fibonacci_sphere_points(4, radius=0.1)
    targets = mics[:1]
    sphere = fibonacci_sphere_points(32, radius=cfg.sphere_radius_a_m)
    kuu, kyy, kuy = build_covariance_blocks(
        mics,
        targets,
        window_w=2,
        ts_s=ts,
        c_sound=cfg.c_sound,
        q=cfg.q_source_intensity,
        sphere_pts=sphere,
        omega1=w1,
        omega2=w2,
    )
    y = np.random.default_rng(0).standard_normal(kyy.shape[0])
    u_hat, _sigma = lmmse_posterior_mean(y, kuu, kuy, kyy, sigma2=1e-3)

    return {
        "kappa_at_zero": round(k0, 4),
        "c00_nonnegative": c00 >= 0.0,
        "u_hat_shape": list(u_hat.shape),
        "kyy_shape": list(kyy.shape),
    }
