"""Equilibrium relaxation and EP contrastive learning (Eq. 3)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.spim_ep.config import SPIMEPConfig
from ltx_trainer.spim_ep.energy import grad_energy_wrt_sm, grad_lambda_k
from ltx_trainer.spim_ep.mattis import MattisParams, build_augmented_j, tilde_coupling
from ltx_trainer.spim_ep.optimizers import bop_step, sgd_step


def relax_to_equilibrium(
    s_init: NDArray[np.floating],
    u: NDArray[np.floating],
    y: NDArray[np.floating] | None,
    params: MattisParams,
    cfg: SPIMEPConfig,
    *,
    beta: float,
    n_steps: int,
) -> tuple[NDArray[np.floating], int]:
    """Gradient descent on E until n_steps (inference / nudged phases)."""
    s = s_init.copy()
    j = build_augmented_j(params.lambdas, params.xi, cfg.n_input)
    j_dyn = j[cfg.n_input :, cfg.n_input :]
    j_tilde = tilde_coupling(j_dyn) if cfg.use_tilde_j else None
    n_spim = 0
    for _ in range(n_steps):
        for m in range(cfg.n_dynamic):
            g = grad_energy_wrt_sm(
                s,
                u,
                params,
                m,
                n_input=cfg.n_input,
                alpha=cfg.alpha,
                beta=beta,
                y=y,
                j=j,
                j_tilde=j_tilde,
                delta=cfg.finite_diff_delta,
                hybrid_digital_input=cfg.hybrid_digital_input,
            )
            s[m] -= cfg.inference_lr * g
            n_spim += 2
        n_spim += 0
    return s, n_spim


def ep_parameter_update(
    params: MattisParams,
    s_free: NDArray[np.floating],
    s_plus: NDArray[np.floating],
    s_minus: NDArray[np.floating],
    cfg: SPIMEPConfig,
    *,
    xi_grad_ema: NDArray[np.floating] | None = None,
) -> tuple[MattisParams, NDArray[np.floating], dict[str, float]]:
    """Contrastive EP update (Eq. 3) for λ (SGD) and ξ (BOP when enabled)."""
    beta = cfg.beta
    k = params.rank
    n = cfg.n_input
    dlam = np.zeros(k, dtype=np.float64)
    dxi = np.zeros_like(params.xi)
    for idx in range(k):
        g_plus = grad_lambda_k(s_plus, params.xi[idx], k, n)
        g_minus = grad_lambda_k(s_minus, params.xi[idx], k, n)
        dlam[idx] = (g_minus - g_plus) / (2.0 * beta)
    for idx in range(k):
        for i in range(params.n_units):
            # approximate via energy difference on xi flip is expensive; use Eq. 8 contrast
            from ltx_trainer.spim_ep.energy import grad_xi_ki

            g_plus = grad_xi_ki(s_plus, params.xi[idx], params.lambdas[idx], i, k, n)
            g_minus = grad_xi_ki(s_minus, params.xi[idx], params.lambdas[idx], i, k, n)
            dxi[idx, i] = (g_minus - g_plus) / (2.0 * beta)
    new_lam = sgd_step(params.lambdas, dlam, cfg.learn_lr_lambda, l2=cfg.l2_lambda)
    if xi_grad_ema is None:
        xi_grad_ema = np.zeros_like(params.xi)
    xi_grad_ema = cfg.bop_gamma * dxi + (1.0 - cfg.bop_gamma) * xi_grad_ema
    if cfg.learn_lr_xi > 0:
        new_xi = params.xi - cfg.learn_lr_xi * dxi
    else:
        new_xi, xi_grad_ema = bop_step(params.xi, xi_grad_ema, cfg.bop_tau)
    return MattisParams(lambdas=new_lam, xi=new_xi), xi_grad_ema, {
        "delta_lambda_norm": float(np.linalg.norm(dlam)),
        "delta_xi_norm": float(np.linalg.norm(dxi)),
    }


def predict_class(s: NDArray[np.floating], n_output: int) -> int:
    """Argmax over output units (one-hot ±1 targets)."""
    out = s[-n_output:]
    return int(np.argmax(out))
