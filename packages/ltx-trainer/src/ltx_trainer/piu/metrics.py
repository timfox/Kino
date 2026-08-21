"""Unlearning evaluation metrics (Supp. A, Eq. 7–10)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def identity_score_matching(
    generated_embeddings: Tensor,
    centroid: Tensor,
) -> float:
    """ISM(k) = (1/n) Σ cos(ArcFace(y_j), μ_k) (Eq. 7)."""
    gen_n = F.normalize(generated_embeddings.float(), dim=-1)
    cen_n = F.normalize(centroid.float().unsqueeze(0), dim=-1)
    sims = (gen_n * cen_n).sum(dim=-1)
    return float(sims.mean().item())


def nearest_centroid_accuracy(
    generated_embeddings: Tensor,
    centroids: dict[int, Tensor],
    true_id: int,
) -> float:
    """Fraction of samples classified to true_id by nearest centroid."""
    if generated_embeddings.numel() == 0:
        return 0.0
    ids = list(centroids.keys())
    stack = torch.stack([centroids[i] for i in ids], dim=0)
    gen_n = F.normalize(generated_embeddings.float(), dim=-1)
    cen_n = F.normalize(stack.float(), dim=-1)
    sims = gen_n @ cen_n.T
    pred_idx = sims.argmax(dim=-1)
    pred_ids = torch.tensor(ids, device=pred_idx.device)[pred_idx]
    return float((pred_ids == true_id).float().mean().item())


def srk_score(
    forget_acc: float,
    retain_acc: float,
    *,
    epsilon: float = 1e-2,
) -> float:
    """SRK = Acc_R / (Acc_U + ε) (Eq. 10); higher is better."""
    return float(retain_acc / (forget_acc + epsilon))


def evaluate_unlearning(
    forget_generated: Tensor,
    retain_generated: Tensor,
    forget_centroid: Tensor,
    retain_centroid: Tensor,
    all_centroids: dict[int, Tensor],
    forget_id: int,
    retain_id: int,
) -> dict[str, float]:
    """ISM + SRK for one forget/retain pair."""
    forget_ism = identity_score_matching(forget_generated, forget_centroid)
    retain_ism = identity_score_matching(retain_generated, retain_centroid)
    acc_u = nearest_centroid_accuracy(forget_generated, all_centroids, forget_id)
    acc_r = nearest_centroid_accuracy(retain_generated, all_centroids, retain_id)
    return {
        "forget_ism": forget_ism,
        "retain_ism": retain_ism,
        "forget_acc": acc_u,
        "retain_acc": acc_r,
        "srk": srk_score(acc_u, acc_r),
    }
