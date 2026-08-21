"""Parametric Memory Law (Eq. 6)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def parametric_memory_law(
    rank: Tensor | float,
    length: Tensor | float,
    *,
    C: float,
    alpha: float,
    beta: float,
    b: float = 0.0,
) -> Tensor:
    """ΔL(r, ℓ) = C · r^α · ℓ^(-β) + b."""
    r = torch.as_tensor(rank, dtype=torch.float32)
    ell = torch.as_tensor(length, dtype=torch.float32)
    return C * torch.pow(r.clamp(min=1e-6), alpha) * torch.pow(ell.clamp(min=1.0), -beta) + b


def loss_from_probability(p: Tensor, eps: float = 1e-8) -> Tensor:
    return -torch.log(p.clamp(min=eps))


def probability_from_loss(loss: Tensor) -> Tensor:
    return torch.exp(-loss)


def delta_loss(l_init: Tensor, l_final: Tensor) -> Tensor:
    return l_init - l_final


def _solve_3x3(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination for 3×3; returns None if singular."""
    m = [row[:] + [bv] for row, bv in zip(a, b, strict=True)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            fac = m[r][col]
            for j in range(col, n + 1):
                m[r][j] -= fac * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_law_grid(
    ranks: list[int],
    lengths: list[int],
    delta_l_grid: list[list[float]],
) -> dict[str, float]:
    """Log-linear OLS: log ΔL ≈ log C + α log r − β log ℓ (b fixed to 0)."""
    rows: list[tuple[int | float, int | float, float]] = []
    for i, r in enumerate(ranks):
        for j, ell in enumerate(lengths):
            d = delta_l_grid[i][j]
            if d <= 0:
                continue
            rows.append((r, ell, d))
    if len(rows) < 3:
        return {"C": 1.0, "alpha": 0.5, "beta": 0.5, "b": 0.0, "r2": 0.0}
    # Normal equations for [1, log r, -log ℓ]
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0, 0.0, 0.0]
    for r, ell, d in rows:
        lr, le, ld = math.log(r), math.log(ell), math.log(d)
        x = [1.0, lr, -le]
        for i in range(3):
            atb[i] += x[i] * ld
            for j in range(3):
                ata[i][j] += x[i] * x[j]
    sol = _solve_3x3(ata, atb)
    if sol is None:
        return {"C": 1.0, "alpha": 0.5, "beta": 0.5, "b": 0.0, "r2": 0.0}
    log_c, alpha, beta = sol
    C = math.exp(log_c)
    b = 0.0
    preds = [
        parametric_memory_law(r, ell, C=C, alpha=alpha, beta=beta, b=b).item()
        for r, ell, _ in rows
    ]
    trues = [d for _, _, d in rows]
    ss_res = sum((p - t) ** 2 for p, t in zip(preds, trues, strict=True))
    ss_tot = sum((t - sum(trues) / len(trues)) ** 2 for t in trues) + 1e-8
    r2 = 1.0 - ss_res / ss_tot
    return {"C": C, "alpha": alpha, "beta": beta, "b": b, "r2": r2}
