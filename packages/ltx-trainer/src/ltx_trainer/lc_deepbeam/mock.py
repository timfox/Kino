"""LC-DeepBeam LCMV + constraint smoke (arXiv:2605.21141)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.lc_deepbeam.config import LcDeepBeamConfig
from ltx_trainer.lc_deepbeam.lcmv import lcmv_weights
from ltx_trainer.lc_deepbeam.losses import pass_penalty, si_sdr


def evaluation_smoke(cfg: LcDeepBeamConfig | None = None) -> dict[str, Any]:
    c = cfg or LcDeepBeamConfig()
    m = c.microphones_m
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    phi = phi @ phi.conj().T / m
    nu = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    nu = nu / (np.linalg.norm(nu) + 1e-12)
    w = lcmv_weights(phi, nu.reshape(-1, 1), np.array([1.0]))
    x = rng.standard_normal(m)
    x_hat = x + 0.05 * rng.standard_normal(m)
    return {
        "paper": c.paper_arxiv,
        "table1_est_rtf_si_sdr_db": c.table1_three_spk_anechoic_est_rtf_si_sdr,
        "lcmv_distortionless": round(abs(float(np.vdot(w, nu))), 4),
        "toy_si_sdr_db": round(si_sdr(x_hat, x), 2),
        "pass_penalty": round(pass_penalty(w, nu), 6),
    }
