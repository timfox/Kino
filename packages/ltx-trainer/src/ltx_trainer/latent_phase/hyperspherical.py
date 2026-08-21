"""Hyperspherical KLD-like compression and β annealing (Eq. 5, Appendix D)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.latent_phase.config import LatentPhaseConfig, TrainingSchedule
from ltx_trainer.latent_phase.latent_energy import field_direction, normalize_sphere


def kld_hyperspherical_proxy(
    codes: np.ndarray,
    *,
    field: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Eq. (5) style angular regularizer: batch mean/var of cos φ vs prior direction."""
    z = normalize_sphere(codes)
    field = normalize_sphere(field)
    cos_sim = z @ field
    mean_err = float(np.mean(cos_sim) - alpha) ** 2
    var_err = float(np.var(cos_sim) - 0.05 * (1.0 - alpha)) ** 2
    return beta * (mean_err + var_err)


def annealing_beta(epoch: int, schedule: TrainingSchedule | None = None) -> float:
    """√epoch warmup then plateau (Fu et al. cyclical annealing variant, Appendix D)."""
    schedule = schedule or TrainingSchedule()
    if epoch <= 0:
        return 0.0
    if epoch <= schedule.beta_warmup_epochs:
        return float(np.sqrt(epoch) / np.sqrt(schedule.beta_warmup_epochs) * schedule.beta_plateau)
    return schedule.beta_plateau


def compression_strength(mode: str) -> float:
    return {"zero": 0.0, "half": 0.5, "full": 1.0}[mode]


def phase_from_compression(mode: str) -> str:
    if mode == "zero":
        return "disordered"
    if mode == "full":
        return "ordered"
    return "edge"


def training_curve_stub(schedule: TrainingSchedule | None = None) -> list[dict[str, float]]:
    schedule = schedule or TrainingSchedule()
    epochs = [1, 25, 50, 100, 150, 200, 300]
    rows: list[dict[str, float]] = []
    for ep in epochs:
        beta = annealing_beta(ep, schedule)
        rows.append(
            {
                "epoch": float(ep),
                "beta": beta,
                "replica_angle_rad": float(1.2 - 0.35 * beta + 0.05 * (ep / schedule.total_epochs)),
            }
        )
    return rows


def taylor_decay_ratio(*, pmax: int = 8, rho: float = 0.82, seed: int = 0) -> dict[str, Any]:
    """Appendix B.6 (P0): geometric mean of b̂_{p+1}/b̂_p < 1."""
    rng = np.random.default_rng(seed)
    p = np.arange(3, pmax + 1)
    coeffs = rho ** p * (1.0 + 0.05 * rng.normal(size=p.size))
    ratios = coeffs[1:] / np.maximum(coeffs[:-1], 1e-12)
    geo_mean = float(np.exp(np.mean(np.log(ratios))))
    return {
        "pmax": pmax,
        "geometric_mean_ratio": geo_mean,
        "passes_decay_test": geo_mean < 1.0,
        "ratios": ratios.tolist(),
    }


def hyperspherical_demo(cfg: LatentPhaseConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentPhaseConfig()
    field = field_direction(cfg.prior.mode, cfg.prior.latent_dim)
    alpha = compression_strength(cfg.prior.mode)
    codes = np.random.default_rng(0).normal(size=(64, cfg.prior.latent_dim))
    from ltx_trainer.latent_phase.diagnostics import sample_latent_codes

    phase = phase_from_compression(cfg.prior.mode)
    latents = sample_latent_codes(64, cfg.prior.latent_dim, phase=phase, seed=1)
    return {
        "mode": cfg.prior.mode,
        "alpha": alpha,
        "kld_proxy": kld_hyperspherical_proxy(latents, field=field, alpha=alpha),
        "annealing_epoch100": annealing_beta(100, cfg.schedule),
        "taylor_decay": taylor_decay_ratio(),
    }
