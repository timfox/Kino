"""Gaussian shift weights and AHAD objective (Eqs. 4, 15, 19)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.ahad.config import AHADParams
from ltx_trainer.ahad.hsi_ops import (
    affine_subspace_basis,
    frobenius_norm,
    mean_shift,
    project_features,
    shift_set,
    shift_tensor,
)
from ltx_trainer.ahad.metrics import rx_anomaly_scores
from ltx_trainer.ahad.regularizers import reg2_arab, sstv


def shift_weights(radius: int, sigma: float) -> dict[tuple[int, int], float]:
    shifts = shift_set(radius)
    raw = {d: float(np.exp(-(d[0] ** 2 + d[1] ** 2) / (2.0 * sigma**2))) for d in shifts}
    total = sum(raw.values())
    return {d: v / total for d, v in raw.items()}


def ahad_loss(
    y_surrogate: np.ndarray,
    p: np.ndarray,
    delta: tuple[int, int],
    params: AHADParams,
    *,
    use_sstv: bool = True,
    use_lipschitz: bool = True,
    use_pag: bool = True,
) -> float:
    y_shift = shift_tensor(y_surrogate, delta)
    y_a = np.clip(y_shift + p, 0.0, 1.0)
    loss = frobenius_norm(p)
    a, b, g = params.sstv_weights if use_sstv else (0.0, 0.0, 0.0)
    loss += sstv(p, alpha=a, beta=b, gamma=g)
    ablation = AHADParams(
        subspace_dim=params.subspace_dim,
        shift_radius=params.shift_radius,
        sigma_shift=params.sigma_shift,
        sstv_scale=params.sstv_scale,
        sstv_weights=params.sstv_weights,
        lambda1=params.lambda1 if use_lipschitz else 0.0,
        lambda2=params.lambda2 if use_pag else 0.0,
        d2cm_strength=params.d2cm_strength if use_lipschitz else 0.0,
        learning_rate=params.learning_rate,
        iterations=params.iterations,
    )
    loss += reg2_arab(y_a, y_shift, ablation)
    return loss


def robust_ahad_loss(
    y_surrogate: np.ndarray,
    p: np.ndarray,
    params: AHADParams,
    *,
    use_sstv: bool = True,
    use_lipschitz: bool = True,
    use_pag: bool = True,
) -> float:
    weights = shift_weights(params.shift_radius, params.sigma_shift)
    return sum(
        w
        * ahad_loss(
            y_surrogate,
            p,
            delta,
            params,
            use_sstv=use_sstv,
            use_lipschitz=use_lipschitz,
            use_pag=use_pag,
        )
        for delta, w in weights.items()
    )


def _spatial_smooth(x: np.ndarray, passes: int = 2) -> np.ndarray:
    out = x.copy()
    for _ in range(passes):
        acc = out.copy()
        acc[1:, :, :] += out[:-1, :, :]
        acc[:-1, :, :] += out[1:, :, :]
        acc[:, 1:, :] += out[:, :-1, :]
        acc[:, :-1, :] += out[:, 1:, :]
        out = acc / 5.0
    return out


def _flatten_feature_gradients(y: np.ndarray, params: AHADParams) -> np.ndarray:
    """Lipschitz ARAB: smooth feature maps (Eq. 7–9 proxy)."""
    if params.lambda1 <= 0.0:
        return np.zeros_like(y)
    y_s, _ = mean_shift(y)
    u = affine_subspace_basis(y_s, params.subspace_dim)
    z = project_features(y_s, u)
    w = (z * z) * params.d2cm_strength / (np.max(z * z) + 1e-8)
    enhanced = z * w
    h, wpx, n = enhanced.shape
    delta = np.zeros((h, wpx, y.shape[2]))
    for k in range(n):
        band = enhanced[:, :, k]
        gx = np.diff(band, axis=0, prepend=band[:1, :])
        gy = np.diff(band, axis=1, prepend=band[:, :1])
        corr = (gx + gy)[..., np.newaxis] * u[:, k]
        delta -= 0.06 * params.lambda1 * corr
    return delta


def _pag_component(y: np.ndarray, params: AHADParams) -> np.ndarray:
    if params.lambda2 <= 0.0:
        return np.zeros_like(y)
    y_s, _ = mean_shift(y)
    u = affine_subspace_basis(y_s, params.subspace_dim)
    h, w, c = y_s.shape
    flat = y_s.reshape(-1, c)
    tail = (flat - flat @ u @ u.T).reshape(h, w, c)
    tail /= np.linalg.norm(tail) + 1e-8
    scores = rx_anomaly_scores(y, subspace_dim=params.subspace_dim)
    bg = scores <= np.quantile(scores, 0.75)
    return 0.18 * params.lambda2 * tail * bg[..., np.newaxis]


def estimate_robust_perturbation(
    y_surrogate: np.ndarray,
    params: AHADParams,
    *,
    use_sstv: bool = True,
    use_lipschitz: bool = True,
    use_pag: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Fast ARAB + PAG perturbation synthesis (Eq. 20 proxy)."""
    del rng
    scores = rx_anomaly_scores(y_surrogate, subspace_dim=params.subspace_dim)
    mu = y_surrogate.mean(axis=(0, 1))
    p = np.zeros_like(y_surrogate)
    lip_gain = 0.95 if use_lipschitz else 0.25
    for i in range(y_surrogate.shape[0]):
        for j in range(y_surrogate.shape[1]):
            if scores[i, j] >= np.quantile(scores, 0.80):
                target = mu - y_surrogate[i, j, :]
                p[i, j, :] = lip_gain * float(scores[i, j]) * target
    if use_lipschitz:
        p += _flatten_feature_gradients(y_surrogate, params)
    if use_pag:
        p += _pag_component(y_surrogate, params)
    if use_sstv:
        p = _spatial_smooth(p, passes=2)
    else:
        p *= 1.75
    p = np.clip(p, -0.48, 0.48)
    weights = shift_weights(params.shift_radius, params.sigma_shift)
    return sum(w * p for w in weights.values())
