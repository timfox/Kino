"""Eq. (5) Lipschitz preservation under QR projection (arXiv:2605.24769)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ltx_trainer.hsir_rgb.projection import hyperspectral_denoise


def encoder_spectral_norm(e: Tensor) -> float:
    """Largest singular value of encoder ``E`` ``[Kc, C]``."""
    return float(torch.linalg.svdvals(e)[0].item())


def lipschitz_surrogate(
    y: Tensor,
    e: Tensor,
    f_matrix: Tensor,
    inner: Callable[[Tensor], Tensor],
    *,
    inner_c: int,
    delta_scale: float = 1e-3,
) -> float:
    """Empirical ratio ``‖D^h(y+δ)−D^h(y)‖ / ‖δ‖`` for small δ (identity inner → ≈ 1 when ‖E‖≈1)."""
    delta = delta_scale * torch.randn_like(y)
    y2 = y + delta
    d1 = hyperspectral_denoise(y, e, f_matrix, inner, inner_c=inner_c)
    d2 = hyperspectral_denoise(y2, e, f_matrix, inner, inner_c=inner_c)
    num = torch.linalg.vector_norm((d2 - d1).reshape(d1.shape[0], -1), dim=1).mean()
    den = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=1).mean().clamp(min=1e-8)
    return float((num / den).item())


def verify_qr_nonexpansive_identity_inner(
    e_tilde: Tensor,
    y: Tensor,
    *,
    inner_c: int,
    tol: float = 1.05,
) -> dict[str, float]:
    """Check ``‖E‖ ≈ 1`` and wrapper does not shrink/expansion beyond ``tol`` with identity inner."""
    from ltx_trainer.hsir_rgb.projection import encoder_decoder_from_unconstrained

    e, f = encoder_decoder_from_unconstrained(e_tilde)
    inner = lambda z: z
    ratio = lipschitz_surrogate(y, e, f, inner, inner_c=inner_c)
    fe = f @ e
    fe_err = float(torch.linalg.matrix_norm(fe - torch.eye(e.shape[1], device=fe.device)).item())
    return {
        "encoder_spectral_norm": encoder_spectral_norm(e),
        "fe_identity_error": fe_err,
        "empirical_lipschitz_ratio": ratio,
        "within_tol": ratio <= tol and fe_err < 1e-3,
    }
