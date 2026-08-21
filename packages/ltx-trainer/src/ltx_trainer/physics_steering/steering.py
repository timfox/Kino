"""Inference-time physics steering (Sec. 3.4, Eq. 5)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.physics_steering.probe import cav_from_weights, classify_binary, logistic_predict_proba


def steer_hidden_states(H: Tensor, cav: Tensor, alpha: float) -> Tensor:
    """Eq. (5): ``H̃_i = H_i + α v`` for all patch tokens."""
    if H.ndim == 2:
        v = cav.reshape(1, -1)
        return H + alpha * v
    if H.ndim == 3:
        v = cav.reshape(1, 1, -1)
        return H + alpha * v
    raise ValueError("H must be (N, D) or (B, N, D)")


def representation_shift(f_before: Tensor, f_after: Tensor) -> Tensor:
    """Δf = f_after − f_before; supports batched (B, D)."""
    return f_after - f_before


def steer_representation(f: Tensor, cav: Tensor, alpha: float) -> Tensor:
    """Steer mean-pooled representation directly (probe-only smoke)."""
    return f + alpha * cav


def probe_impossible_score(
    f: Tensor,
    weight: Tensor,
    bias: float | Tensor,
) -> Tensor:
    """P(impossible) for (D,) or (B, D)."""
    if f.ndim == 1:
        return logistic_predict_proba(f, weight, bias)
    return logistic_predict_proba(f, weight, bias)


def steer_batch_scores(
    features: Tensor,
    weight: Tensor,
    bias: float | Tensor,
    cav: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return baseline prob, steered prob, and binary preds after steering."""
    base = probe_impossible_score(features, weight, bias)
    steered_f = steer_representation(features, cav, alpha)
    steered = probe_impossible_score(steered_f, weight, bias)
    return base, steered, classify_binary(steered)


def project_out_direction(features: Tensor, direction: Tensor) -> Tensor:
    """Remove component along unit ``direction`` for iterative orthogonal probes."""
    v = direction / (direction.norm() + 1e-8)
    if features.ndim == 1:
        coeff = features @ v
        return features - coeff * v
    coeff = (features * v).sum(dim=-1, keepdim=True)
    return features - coeff * v


def iterative_orthogonal_probe_accuracies(
    features: Tensor,
    labels: Tensor,
    *,
    max_iters: int = 5,
    fit_steps: int = 200,
) -> list[float]:
    """Fig. 2: fit probe, project out CAV, repeat."""
    from ltx_trainer.physics_steering.probe import fit_logistic_probe

    residual = features.clone()
    accs: list[float] = []
    for _ in range(max_iters):
        w, _, acc = fit_logistic_probe(residual, labels, steps=fit_steps)
        accs.append(acc)
        v = cav_from_weights(w)
        residual = torch.stack([project_out_direction(residual[i], v) for i in range(residual.shape[0])])
    return accs


def angle_between(a: Tensor, b: Tensor) -> float:
    """Degrees between unit directions."""
    a_n = a / (a.norm() + 1e-8)
    b_n = b / (b.norm() + 1e-8)
    cos = float(torch.clamp(a_n @ b_n, -1.0, 1.0).item())
    return math.degrees(math.acos(cos))
