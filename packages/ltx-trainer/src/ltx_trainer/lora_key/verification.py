"""Ownership verification via hypothesis testing — § IV-D, Eqs. 15–18."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def matching_bits(registered: Tensor, recovered: Tensor) -> int:
    """M(m, m') — Hamming agreement on L bits."""
    reg = (registered > 0.5).int()
    rec = (recovered > 0.5).int()
    return int((reg == rec).sum().item())


def binomial_fpr(L: int, tau: int) -> float:
    """P(M > τ | H0) under B(L, 0.5) — Eq. 17."""
    if tau >= L:
        return 0.0
    p = 0.0
    for i in range(tau + 1, L + 1):
        p += math.comb(L, i) * (0.5**L)
    return p


def verification_threshold(L: int, *, target_fpr: float = 1e-6) -> int:
    """Smallest τ with FPR(τ) ≤ target_fpr."""
    for tau in range(L, -1, -1):
        if binomial_fpr(L, tau) <= target_fpr:
            return tau
    return L


def verify_ownership(
    registered: Tensor,
    recovered: Tensor,
    *,
    target_fpr: float = 1e-6,
) -> dict[str, Any]:
    """Reject H0 if M(m, m') > τ_ver (Eq. 16)."""
    L = int(registered.numel())
    tau = verification_threshold(L, target_fpr=target_fpr)
    matches = matching_bits(registered, recovered)
    accepted = matches > tau
    return {
        "accepted": accepted,
        "matching_bits": matches,
        "message_bits": L,
        "threshold_tau": tau,
        "fpr_at_threshold": binomial_fpr(L, tau),
        "bit_accuracy": matches / max(L, 1),
    }
