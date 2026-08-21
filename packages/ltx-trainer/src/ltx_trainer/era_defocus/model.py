"""ErA unrolled network stub — Sec. 3.3."""

from __future__ import annotations

from typing import Any

from ltx_trainer.era_defocus.config import EraDefocusConfig
from ltx_trainer.era_defocus.unrolling import (
    alm_multipliers,
    cnn_denoise_stub,
    update_e,
    update_u,
    update_x_fft_stub,
)


def basis_weight_psf_stub(length: int, *, n_basis: int = 4) -> tuple[list[float], list[float]]:
    """Global kernel basis + per-pixel weight map (stub)."""
    basis = [1.0 / (i + 1.0) for i in range(n_basis)]
    weights = [1.0 / length] * length
    kernel = [sum(b * w for b, w in zip(basis, weights[:n_basis])) / n_basis] * length
    s = sum(kernel) or 1.0
    return [k / s for k in kernel], weights


def era_forward_stub(
    obs: list[float],
    cfg: EraDefocusConfig | None = None,
) -> dict[str, Any]:
    """Run K ALM unrolling iterations on 1-D toy signal."""
    cfg = cfg or EraDefocusConfig()
    kernel, _weights = basis_weight_psf_stub(len(obs))
    x = list(obs)
    u = obs[0]
    e = 0.0
    z = x[0]
    p = 0.0
    gamma = omega = delta = 0.0

    for _ in range(cfg.unrolling_depth):
        h_conv = sum(k * v for k, v in zip(kernel, x)) / len(x)
        u = update_u(h_conv, gamma, obs[0], e, lambda1=cfg.lambda1)
        x_val = update_x_fft_stub(
            u,
            gamma,
            omega,
            z,
            lambda1=cfg.lambda1,
            lambda2=cfg.lambda2,
        )
        x = [x_val] * len(x)
        e = update_e(u, obs[0], delta, p, lambda3=cfg.lambda3)
        z = cnn_denoise_stub(x[0] + omega / cfg.lambda2)
        p = cnn_denoise_stub(e - delta / cfg.lambda3)
        gamma, omega, delta = alm_multipliers(
            gamma,
            omega,
            delta,
            h_conv_x=h_conv,
            u=u,
            x=x[0],
            z=z,
            e=e,
            p=p,
            lambda1=cfg.lambda1,
            lambda2=cfg.lambda2,
            lambda3=cfg.lambda3,
        )

    return {"pred": x, "kernel": kernel, "error_e": e, "final_u": u}
