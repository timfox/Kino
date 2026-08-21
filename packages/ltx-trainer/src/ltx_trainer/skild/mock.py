"""SKILD k-space diffusion smoke (arXiv:2605.26032)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.skild.config import SkildConfig
from ltx_trainer.skild.dct import frequency_radius_grid
from ltx_trainer.skild.spectrum import fit_power_law_spectrum, radial_variance_spectrum


def evaluation_smoke() -> dict[str, Any]:
    cfg = SkildConfig()
    h, w = 32, 32
    grid = frequency_radius_grid(h, w)
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((h, w))
    centers, var = radial_variance_spectrum([patch])
    fit = fit_power_law_spectrum(centers, var)
    from ltx_trainer.skild.inference import start_timestep_for_scale

    t0 = start_timestep_for_scale(4.0, h, w, cfg=cfg)
    return {
        "paper": "arXiv:2605.26032",
        "grid_max_radius": round(float(grid.max()), 3),
        "spectrum_bins": len(centers),
        "power_law_exponent": round(float(fit.exponent), 4),
        "start_timestep_x4": int(t0),
    }
