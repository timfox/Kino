"""Mutual-information objectives for shared alignment and unique disentanglement (Eq. 6–7, 10)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    a_n = F.normalize(a.unsqueeze(0), dim=-1)
    b_n = F.normalize(b.unsqueeze(0), dim=-1)
    return (a_n @ b_n.T).squeeze()


def info_nce_shared(z_i: Tensor, z_j_pos: Tensor, z_j_neg_bank: Tensor, *, temperature: float = 0.07) -> Tensor:
    r"""Lower-bound style shared MI (Eq. 6): contrastive match between same-anchor flows."""
    pos = cosine_sim(z_i, z_j_pos) / temperature
    negs = torch.stack([cosine_sim(z_i, z_n) / temperature for z_n in z_j_neg_bank], dim=0)
    logits = torch.cat([pos.unsqueeze(0), negs], dim=0)
    labels = torch.zeros(1, dtype=torch.long, device=z_i.device)
    return F.cross_entropy(logits.unsqueeze(0), labels)


def nce_club_unique_upper_bound(
    z_i: Tensor,
    z_j_pos: Tensor,
    z_j_neg: Tensor,
    critic: callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    r"""NCE-CLUB upper bound on unique MI — minimize to disentangle (Eq. 7).

    Uses a simple bilinear critic ``f(x,y) = (x·y).sum()`` when ``critic`` is None.
    """
    if critic is None:
        critic = lambda x, y: (x * y).sum()

    pos = critic(z_i, z_j_pos)
    neg = critic(z_i, z_j_neg)
    return pos - neg


def directed_shared_alignment(
    z_src: Tensor,
    z_tgt: Tensor,
    z_tgt_neg_bank: Tensor,
    *,
    temperature: float = 0.07,
    stop_grad_target: bool = True,
) -> Tensor:
    r"""Asymmetric InfoNCE with stop-gradient (Eq. 10): ``L_{U→G}`` or ``L_{G→U}``."""
    tgt = z_tgt.detach() if stop_grad_target else z_tgt
    pos = cosine_sim(z_src, tgt) / temperature
    negs = torch.stack([cosine_sim(z_src, z_n.detach() if stop_grad_target else z_n) / temperature for z_n in z_tgt_neg_bank], dim=0)
    logits = torch.cat([pos.unsqueeze(0), negs], dim=0)
    labels = torch.zeros(1, dtype=torch.long, device=z_src.device)
    return F.cross_entropy(logits.unsqueeze(0), labels)


def stage2_total_loss(
    l_u2g: Tensor,
    l_g2u: Tensor,
    l_uni: Tensor,
    l_und: Tensor,
    l_gen: Tensor,
    *,
    lambda_uni: float = 0.6,
) -> Tensor:
    """``L_total = L_{U→G} + L_{G→U} + L_uni + L_und + L_gen`` (Eq. 11)."""
    return l_u2g + l_g2u + lambda_uni * l_uni + l_und + l_gen
