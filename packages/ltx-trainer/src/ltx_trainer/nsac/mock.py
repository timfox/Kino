"""NSAC evaluation smoke — OU moments + stochastic training loss (arXiv:2605.26061)."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.smoke_util import load_sibling


def _ou_numpy(kappa: float, phi: float, psi: float, t: float, a0: float = 0.0) -> tuple[float, float]:
    exp_mkt = math.exp(-kappa * t)
    mean = phi + (a0 - phi) * exp_mkt
    var = (psi * psi) / (2.0 * kappa) * (1.0 - math.exp(-2.0 * kappa * t))
    return mean, var


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg_mod = load_sibling(__file__, "config")
    cfg = cfg or cfg_mod.NSACConfig(d_model=32, n_heads=4, n_mc=2, lambda_reg=0.05, top_k=8)
    mean, var = _ou_numpy(1.0, 0.5, 0.2, 0.5)

    out: dict[str, Any] = {
        "paper": "arXiv:2605.26061",
        "ou_mean": round(mean, 4),
        "ou_var": round(var, 6),
        "lambda_reg": cfg.lambda_reg,
    }

    try:
        import torch

        ou_mod = load_sibling(__file__, "ou")
        losses_pkg = load_sibling(__file__, "losses")
        module_mod = load_sibling(__file__, "module")

        torch.manual_seed(0)
        kappa = torch.tensor([1.0, 2.0])
        phi = torch.tensor([0.5, -0.25])
        psi = torch.tensor([0.2, 0.3])
        t = torch.tensor([0.5])
        a0 = torch.zeros(1)
        mean_t, var_t = ou_mod.ou_mean_variance(a0, kappa, phi, psi, t, kappa_floor=cfg.kappa_floor)
        out["ou_mean"] = round(float(mean_t[0]), 4)
        out["ou_var"] = round(float(var_t[0]), 6)

        y = torch.randn(4, 1)
        mu = torch.randn(4, 1)
        log_std = torch.zeros(4, 1)
        out["gaussian_nll_toy"] = round(float(losses_pkg.gaussian_nll(y, mu, log_std).item()), 4)

        model = module_mod.NSACRegressor(in_dim=3, d_out=1, cfg=cfg)
        loss_fn = losses_pkg.NSACTrainingLoss(
            n_mc=cfg.n_mc,
            lambda_reg=cfg.lambda_reg,
            mu_pert=cfg.mu_pert,
            sigma_pert=cfg.sigma_pert,
            reg_eps=cfg.reg_eps,
        )
        x = torch.randn(4, 10, 3)
        y_train = torch.randn(4, 1)
        total, parts = loss_fn(model, x, y_train)
        out["lnll"] = round(float(parts["lnll"]), 4)
        out["lreg"] = round(float(parts["lreg"]), 4)
        out["training_total_finite"] = bool(torch.isfinite(total))
    except ImportError:
        pass

    return out
