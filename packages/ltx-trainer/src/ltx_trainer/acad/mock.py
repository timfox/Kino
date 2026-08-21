"""ACAD smoke evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.acad.config import AcadConfig
from ltx_trainer.acad.losses import joint_loss, si_snr_db
from ltx_trainer.acad.masking import contextual_denoising_mask, masked_magnitude


def evaluation_smoke(cfg: AcadConfig | None = None, seed: int = 0) -> dict[str, object]:
    cfg = cfg or AcadConfig()
    rng = np.random.default_rng(seed)
    t = 256
    clean = rng.normal(0, 0.3, t)
    noise = rng.normal(0, 0.5, t)
    noisy_mag = np.abs(clean + noise) + 0.1
    e = np.array([noisy_mag.mean(), noisy_mag.std(), float(rng.random())])
    mask = contextual_denoising_mask(noisy_mag, e)
    est_mag = masked_magnitude(noisy_mag, mask)
    est = est_mag / (np.max(est_mag) + 1e-8) * np.sign(clean + noise)
    snr = si_snr_db(est, clean)
    l_tot = joint_loss(l_asc=0.2, l_den=0.5)

    return {
        "toy_si_snr_db": float(round(snr, 3)),
        "joint_loss_toy": float(round(l_tot, 4)),
        "paper_unet_si_sdr_db": cfg.unet_si_sdr_db,
        "paper_unet_tu_asc_si_sdr_db": cfg.unet_tu_asc_si_sdr_db,
        "asc_test_accuracy_pct": cfg.asc_test_accuracy_pct,
    }
