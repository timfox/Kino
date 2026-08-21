"""PhyWorld-style training helpers for LTX Phase 2 (fold sidecars + DPO surrogate).

Uses preprocess ``phyworld.physics_proxy`` (1–5) from ``fold.py``. Full diffusion DPO
needs winner/loser video pairs; here we add a batch ranking surrogate when B>1 and
reinforce the existing downweighting path in ``av_fold_training``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.phyworld.dpo import preference_margin_ok


def physics_ranking_preference_loss(
    per_sample_loss: Tensor,
    physics_proxy: Tensor,
    *,
    beta: float = 2.0,
    min_proxy_margin: float = 0.25,
) -> Tensor | None:
    """Pairwise surrogate: higher ``physics_proxy`` should have lower flow loss.

    Uses a reference-free diffusion-DPO logit:
    ``β * (loss_loser - loss_winner)``. The higher-physics sample is the winner,
    so the term is small only when it has lower flow loss. Returns a scalar loss
    added to the batch loss, or ``None`` if B<2 or no valid pairs.
    """
    if per_sample_loss.ndim != 1 or physics_proxy.ndim != 1:
        return None
    b = int(per_sample_loss.shape[0])
    if b < 2:
        return None

    loss = per_sample_loss.float()
    proxy = physics_proxy.detach().float()
    total = per_sample_loss.new_tensor(0.0)
    n_pairs = 0

    for i in range(b):
        for j in range(i + 1, b):
            if proxy[i] >= proxy[j]:
                winner, loser = i, j
                pw, pl = float(proxy[i]), float(proxy[j])
            else:
                winner, loser = j, i
                pw, pl = float(proxy[j]), float(proxy[i])
            if not preference_margin_ok(pw, pl, min_margin=min_proxy_margin):
                continue
            logit = beta * (loss[loser] - loss[winner])
            total = total + F.softplus(-logit)
            n_pairs += 1

    if n_pairs == 0:
        return None
    return total / n_pairs


def physics_post_train_filter(
    physics_proxy: Tensor,
    *,
    min_proxy: float = 3.25,
) -> Tensor:
    """Boolean mask for generated/eval clips that pass the fold-sidecar physics judge."""
    return physics_proxy.float() >= min_proxy


def physics_proxy_regularizer(
    physics_proxy: Tensor,
    *,
    target: float = 4.0,
    scale: float = 0.02,
) -> Tensor:
    """Soft pull: encourage batches toward high physics_proxy (judge overall scale)."""
    return scale * (target - physics_proxy.float().mean()).pow(2)
