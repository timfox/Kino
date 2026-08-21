"""NC-PSZ neighbor-consistency smoke (arXiv:2605.21891)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.nc_psz.config import NcPszConfig
from ltx_trainer.nc_psz.losses import (
    baseline_psz_loss_toy,
    neighbor_consistency_loss,
    same_region_mask,
    total_training_loss,
)
from ltx_trainer.nc_psz.metrics import isolation_ratio_db, neighborhood_median


def evaluation_smoke(cfg: NcPszConfig | None = None) -> dict[str, Any]:
    c = cfg or NcPszConfig()
    izi_vals = np.array([5.2, 5.8, 6.1, 6.0, 5.9], dtype=np.float64)
    g = np.ones(8, dtype=np.float64)
    l_nc = neighbor_consistency_loss(g, g * 0.95, mask=True)
    l_psz = baseline_psz_loss_toy(bright_error=0.01, dark_energy=0.02)
    l_total = total_training_loss(l_psz, l_nc, lam=c.nc_lambda_woofer)
    x1 = np.array([-0.40, 1.10])
    x = np.array([-0.40, 1.10, 0.50, 1.05])
    x_prime = np.array([-0.40, 1.10, 0.51, 1.06])
    return {
        "paper": c.paper_arxiv,
        "izi_median_db": neighborhood_median(izi_vals),
        "paper_woofer_ipi_sigma_rms_imp_pct": c.woofer_ipi_sigma_rms_imp_pct,
        "neighbor_consistency": round(l_nc, 6),
        "total_loss": round(l_total, 4),
        "same_region": same_region_mask(x, x_prime, x1, dov=c.overlap_threshold_m),
        "izi_ratio_db": round(isolation_ratio_db(1.0, 0.1), 2),
    }
