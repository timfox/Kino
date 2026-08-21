"""Multimodal evidence construction (Sec. 4.1, Eq. 2–3)."""

from __future__ import annotations

import torch
from torch import Tensor


def rank_normalize(x: Tensor) -> Tensor:
    """Rank-normalize scores within the last dimension to [0, 1]."""
    if x.numel() == 0:
        return x
    ranks = x.argsort(dim=-1).argsort(dim=-1).to(dtype=x.dtype)
    denom = max(int(x.shape[-1]) - 1, 1)
    return ranks / denom


def build_multimodal_evidence(
    sv: Tensor,
    sa: Tensor,
    scob: Tensor,
    sqv: Tensor,
    sqa: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return routed visual, audio, aligned evidence and base score B(t) (Eq. 2–4).

    All inputs are per-timestep scores with shape (..., T).
    """
    sv_n = rank_normalize(sv)
    sa_n = rank_normalize(sa)
    scob_n = rank_normalize(scob)
    sqv_n = rank_normalize(sqv)
    sqa_n = rank_normalize(sqa)

    sqa_hat = sqa_n * torch.maximum(sa_n, scob_n)
    ev = torch.maximum(sqv_n, sv_n)
    ea = torch.maximum(sqa_hat, sa_n)

    eav_sem = torch.minimum(sqv_n, sqa_hat)
    eav_burst = torch.minimum(torch.minimum(ev, ea), scob_n)
    eav = torch.maximum(eav_sem, eav_burst)

    ev_hat = torch.relu(ev - eav)
    ea_hat = torch.relu(ea - eav)
    base = torch.maximum(torch.maximum(ev_hat, ea_hat), eav)
    return ev_hat, ea_hat, eav, base
