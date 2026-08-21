"""Multi-Prototype Visual-Textual Confidence (Sec. 3.3, Eq. 4–8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.cmap.config import CMAPConfig
from ltx_trainer.cmap.kmeans import kmeans_torch
from ltx_trainer.cmap.routing import cosine_sim


def fit_visual_prototypes_per_class(
    train_visual: Tensor,
    train_labels: Tensor,
    num_classes: int,
    k: int,
    *,
    kmeans_seed: int | None = 0,
) -> Tensor:
    """Return ``P`` (C, K, d) cluster centroids per class from training features ``train_visual`` (N, d)."""
    d = train_visual.shape[-1]
    p = torch.zeros(num_classes, k, d, device=train_visual.device, dtype=train_visual.dtype)
    for c in range(num_classes):
        m = train_labels == c
        ni = int(m.sum().item())
        if ni < 1:
            continue
        xc = train_visual[m]
        if ni < k:
            p[c, :ni] = xc
            mu = xc.mean(dim=0, keepdim=True)
            p[c, ni:] = mu.expand(k - ni, -1)
            continue
        centers, _ = kmeans_torch(xc, k, seed=kmeans_seed + c if kmeans_seed is not None else None)
        p[c] = centers
    return p


def class_visual_confidence(v: Tensor, prototypes: Tensor) -> Tensor:
    """Eq. (4): max over k of sim(v, p_{c,k}); ``v`` (B, d), ``prototypes`` (C, K, d) → (B, C)."""
    v_n = F.normalize(v, dim=-1)
    p_n = F.normalize(prototypes, dim=-1)
    sims = torch.einsum("bd,ckd->bck", v_n, p_n).max(dim=-1).values
    return sims


def class_textual_confidence(v: Tensor, class_text: Tensor) -> Tensor:
    """Eq. (5): ``sim(v, e_c)``; ``class_text`` (C, d)."""
    return cosine_sim(v.unsqueeze(1), class_text.unsqueeze(0))


def joint_class_confidence(v: Tensor, prototypes: Tensor, class_text: Tensor) -> Tensor:
    """Eq. (6): average of visual max-sim and textual sim → (B, C)."""
    return 0.5 * class_visual_confidence(v, prototypes) + 0.5 * class_textual_confidence(v, class_text)


def task_confidence_from_joint(
    joint: Tensor,
    class_mask: Tensor,
    *,
    topk: int = 5,
) -> Tensor:
    """Eq. (7): mean of top-``k`` scores among classes in the selected task.

    ``joint`` (B, C_all), ``class_mask`` (C_all,) bool — True for classes in Y_{t*}.
    """
    masked = joint.masked_fill(~class_mask.unsqueeze(0), float("-inf"))
    k = min(topk, int(class_mask.sum().item()))
    if k < 1:
        return torch.zeros(joint.shape[0], device=joint.device, dtype=joint.dtype)
    topv, _ = masked.topk(k, dim=-1)
    return topv.mean(dim=-1)


def calibrate_task_thresholds(
    confidences: Tensor,
    cfg: CMAPConfig,
) -> tuple[Tensor, Tensor]:
    """Return scalar θ_up, θ_low from 1D training confidences (Sec. 3.3)."""
    if confidences.numel() == 0:
        z = torch.zeros((), device=confidences.device, dtype=confidences.dtype)
        return z + 0.8, z + 0.2
    qu = torch.quantile(confidences.float(), cfg.percentile_high)
    ql = torch.quantile(confidences.float(), cfg.percentile_low)
    return qu, ql


def prompting_weight(conf: Tensor, theta_up: Tensor, theta_low: Tensor) -> Tensor:
    """Eq. (8): piecewise weight ``w(x)`` for batch confidences ``conf`` (B,).

    Returns ``1.0`` if ``conf > θ_up``, ``0.0`` if ``conf < θ_low``, else ``conf``.
    """
    return torch.where(
        conf > theta_up,
        torch.ones_like(conf),
        torch.where(conf < theta_low, torch.zeros_like(conf), conf),
    )
