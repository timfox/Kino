"""FedAvg and gradient alignment (Eqs. 1, 6–8)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.btscafe.config import BTSCafeConfig


def fedavg_aggregate(client_weights: list[float], client_params: list[np.ndarray]) -> np.ndarray:
    """Eq. 1 — weighted parameter aggregation."""
    if not client_params:
        raise ValueError("client_params must be non-empty")
    total = float(sum(client_weights))
    out = np.zeros_like(client_params[0], dtype=np.float64)
    for w, p in zip(client_weights, client_params, strict=True):
        out += (w / total) * p
    return out


def global_reference_gradient(client_grads: list[np.ndarray]) -> np.ndarray:
    """Eq. 6 — mean client gradient."""
    if not client_grads:
        raise ValueError("client_grads must be non-empty")
    return np.mean(np.stack(client_grads, axis=0), axis=0)


def alignment_penalty(client_grad: np.ndarray, ref_grad: np.ndarray, num_params: int) -> float:
    """Eq. 7 — squared L2 discrepancy normalized by parameter count."""
    diff = client_grad - ref_grad
    return float(np.sum(diff * diff) / max(num_params, 1))


def local_loss(
    loss_non_aug: float,
    loss_aug: float,
    align_penalty: float,
    *,
    round_idx: int,
    cfg: BTSCafeConfig | None = None,
) -> dict[str, Any]:
    """Eq. 8 — curriculum local objective with GIN and alignment gates."""
    cfg = cfg or BTSCafeConfig()
    aug_on = int(round_idx > cfg.t_aug)
    align_on = int(round_idx > cfg.t_w)
    total = loss_non_aug + aug_on * loss_aug + align_on * cfg.lambda_align * align_penalty
    return {
        "loss": total,
        "loss_non_aug": loss_non_aug,
        "loss_aug": loss_aug if aug_on else 0.0,
        "align_penalty": align_penalty if align_on else 0.0,
        "gin_active": bool(aug_on),
        "alignment_active": bool(align_on),
    }


def gradient_alignment_smoke(
    num_clients: int = 3,
    param_dim: int = 16,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Toy single-sample gradient alignment across clients."""
    rng = np.random.default_rng(seed)
    grads = [rng.standard_normal(param_dim) for _ in range(num_clients)]
    ref = global_reference_gradient(grads)
    penalties = [alignment_penalty(g, ref, param_dim) for g in grads]
    return {
        "num_clients": num_clients,
        "reference_norm": float(np.linalg.norm(ref)),
        "mean_align_penalty": float(np.mean(penalties)),
        "penalties": penalties,
    }
