"""Logistic probes and CAV extraction (Sec. 3.3, Eq. 4)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def logistic_predict_proba(f: Tensor, weight: Tensor, bias: float | Tensor) -> Tensor:
    """P(impossible=1 | f) with ``weight`` (D,) pointing toward impossible."""
    logits = f @ weight + bias
    return torch.sigmoid(logits)


def cav_from_weights(weight: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Eq. (4): ``v = w / ||w||``; +v → impossible, −v → possible."""
    return weight / (weight.norm() + eps)


def direction_flip_if_needed(
    weight: Tensor,
    bias: Tensor,
    accuracy: float,
) -> tuple[Tensor, Tensor, float]:
    """Sec. 3.3: if acc < 0.5, negate w and b and report 1 − acc."""
    if accuracy >= 0.5:
        return weight, bias, accuracy
    return -weight, -bias, 1.0 - accuracy


def fit_logistic_probe(
    features: Tensor,
    labels: Tensor,
    *,
    c: float = 1.0,
    steps: int = 200,
    lr: float = 0.05,
) -> tuple[Tensor, Tensor, float]:
    """Binary logistic regression on mean-pooled features (possible=0, impossible=1).

    Returns ``(weight, bias, train_accuracy)``. Uses torch SGD; sklearn L-BFGS is external.
    """
    if features.ndim != 2:
        raise ValueError("features must be (N, D)")
    n, d = features.shape
    w = torch.zeros(d, device=features.device, dtype=features.dtype, requires_grad=True)
    b = torch.zeros((), device=features.device, dtype=features.dtype, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    y = labels.float().reshape(-1)
    for _ in range(steps):
        opt.zero_grad()
        logits = features @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y) + 0.5 * c * (w.pow(2).mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (logistic_predict_proba(features, w, b) >= 0.5).float()
        acc = float((pred == y).float().mean().item())
    w_out, b_out, acc_out = direction_flip_if_needed(w.detach(), b.detach(), acc)
    return w_out, b_out, acc_out


def pca_reduce(features: Tensor, n_components: int) -> tuple[Tensor, Tensor]:
    """Center + truncated SVD → (X_reduced, V_pca) with ``X ≈ X_reduced @ V_pca``."""
    if features.ndim != 2:
        raise ValueError("features must be (N, D)")
    n, d = features.shape
    k = min(n_components, n, d)
    x = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    v = vh[:k]
    reduced = x @ v.T
    return reduced, v


def map_pca_weights_to_full(w_reduced: Tensor, v_pca: Tensor) -> Tensor:
    """``w_full = V_pca^T w_reduced`` (Sec. 3.3)."""
    return v_pca.T @ w_reduced


def fit_probe_with_pca(
    features: Tensor,
    labels: Tensor,
    *,
    n_components: int = 64,
    **fit_kwargs: float | int,
) -> tuple[Tensor, Tensor, float]:
    reduced, v = pca_reduce(features, n_components)
    w_r, b, acc = fit_logistic_probe(reduced, labels, **fit_kwargs)
    w = map_pca_weights_to_full(w_r, v)
    return w, b, acc


def classify_binary(prob_impossible: Tensor) -> Tensor:
    return (prob_impossible >= 0.5).long()
