"""Fuzzy memberships, category credibility, and uncertainty (Sec. III-B, Eq. 2–4, 8–10)."""

from __future__ import annotations

import math
from typing import Sequence


def _relu(x: float) -> float:
    return max(0.0, x)


def lp_normalize(logits: Sequence[float], p: int = 2) -> list[float]:
    r"""``a / ||a||_p`` with safe denominator (Eq. 8 step 1)."""
    vals = [float(x) for x in logits]
    if p == 1:
        norm = sum(abs(v) for v in vals)
    else:
        norm = math.sqrt(sum(v * v for v in vals))
    if norm < 1e-12:
        return [0.0] * len(vals)
    return [v / norm for v in vals]


def logits_to_memberships(logits: Sequence[float], *, p: int = 2) -> list[float]:
    r"""``m = ReLU(a / ||a||_p)`` (Eq. 8)."""
    normed = lp_normalize(logits, p=p)
    return [_relu(v) for v in normed]


def necessity_from_membership(m: Sequence[float], k: int) -> float:
    r"""``e_k = 1 - max_{l≠k} m_l`` (Eq. 2)."""
    mk = list(m)
    others = [mk[i] for i in range(len(mk)) if i != k]
    return 1.0 - max(others) if others else 1.0


def category_credibility(m: Sequence[float]) -> list[float]:
    r"""``c_k = (m_k + e_k) / 2`` for all categories (Eq. 3)."""
    mk = list(m)
    out: list[float] = []
    for k in range(len(mk)):
        e_k = necessity_from_membership(mk, k)
        out.append(0.5 * (mk[k] + e_k))
    return out


def credibility_entropy_component(c: float) -> float:
    r"""``H(c) = -c ln c - (1-c) ln(1-c)`` per category (Eq. 4)."""
    c = min(max(float(c), 1e-12), 1.0 - 1e-12)
    return -(c * math.log(c) + (1.0 - c) * math.log(1.0 - c))


def uncertainty_from_credibility(c: Sequence[float]) -> float:
    r"""Normalized mean binary entropy over categories (Eq. 4), in ``[0, 1]``."""
    k = len(c)
    if k == 0:
        return 0.0
    h_sum = sum(credibility_entropy_component(ci) for ci in c)
    return h_sum / (k * math.log(2.0))


def training_credibility(m: Sequence[float], y: Sequence[float]) -> list[float]:
    r"""Training-phase credibility ``r`` (Eq. 10)."""
    mk = list(m)
    yk = [float(v) for v in y]
    k_n = len(mk)
    l_star = max(range(k_n), key=lambda i: yk[i])
    out: list[float] = []
    for k in range(k_n):
        if yk[k] >= 0.5:
            e_k = necessity_from_membership(mk, k)
            out.append(0.5 * (mk[k] + e_k))
        else:
            out.append(0.5 * (mk[k] + (1.0 - mk[l_star])))
    return out


def gamma_schedule(epoch: int, warmup_epochs: int) -> float:
    r"""``γ = min(t / N_w, 1)`` for fused loss warm-up (Eq. 11)."""
    if warmup_epochs <= 0:
        return 1.0
    return min(epoch / warmup_epochs, 1.0)
