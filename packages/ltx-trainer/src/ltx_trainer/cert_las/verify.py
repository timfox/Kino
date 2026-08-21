"""Watermark Robustness (WR), Reference Probability (RP), and ownership test (Cert-LAS §4.5)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ltx_trainer.cert_las.stats import phi_inv


def t_quantile_one_sided(alpha: float, df: int) -> float:
    """(1-alpha) quantile of t(df); normal fallback for large df."""
    df = max(1, int(df))
    if df >= 60:
        # one-sided 95% ~ 1.645; scale by alpha
        z = phi_inv(1.0 - alpha)
        return z
    table = {
        1: 6.314,
        2: 2.920,
        5: 2.015,
        10: 1.812,
        20: 1.725,
        30: 1.697,
        50: 1.676,
        100: 1.660,
    }
    keys = sorted(table.keys())
    pick = keys[0]
    for k in keys:
        if df >= k:
            pick = k
    return table[pick]


def empirical_wr_rp(
    suspect_hits: Sequence[float],
    reference_hits: Sequence[float],
) -> tuple[float, float]:
    """WR and RP as mean Bernoulli rates (already noise-averaged per sample)."""
    s = np.asarray(suspect_hits, dtype=np.float64)
    r = np.asarray(reference_hits, dtype=np.float64)
    wr = float(s.mean()) if s.size else 0.0
    rp = float(r.mean()) if r.size else 0.0
    return wr, rp


def ownership_threshold(
    *,
    M: int,
    N: int,
    zeta: float,
    alpha: float = 0.05,
) -> float:
    """
    Theorem 4.8 / Eq. (5): closed-form WR threshold given RP ≤ ζ.
    """
    M = max(1, int(M))
    N = max(2, int(N))
    zeta = float(min(max(zeta, 0.0), 1.0))
    t2 = t_quantile_one_sided(alpha, N - 1) ** 2
    mn = M * N
    gamma = (2 * mn * zeta + t2) ** 2 - (4 * mn + t2) * (
        mn * zeta**2 - t2 * zeta + t2 * zeta**2
    )
    gamma = max(0.0, gamma)
    return (2 * mn * zeta + t2 + math.sqrt(gamma)) / (2 * mn + t2)


def verify_ownership(
    wr: float,
    rp: float,
    *,
    M: int,
    N: int,
    alpha: float = 0.05,
    zeta: float | None = None,
) -> dict[str, float | bool]:
    """Ownership if WR exceeds closed-form threshold with RP bounded by zeta."""
    zeta_bound = float(rp if zeta is None else zeta)
    tau = ownership_threshold(M=M, N=N, zeta=zeta_bound, alpha=alpha)
    gap = wr - rp
    return {
        "wr": float(wr),
        "rp": float(rp),
        "wr_rp_gap": float(gap),
        "threshold_tau": float(tau),
        "verified": bool(wr > tau),
        "rp_upper_zeta": zeta_bound,
    }


def simulate_wr_rp(
    *,
    p_suspect: float,
    p_ref: float,
    M: int,
    N: int,
    seed: int = 0,
) -> tuple[float, float]:
    """Monte Carlo WR/RP under layer-adaptive noise averaging (Eq. 4)."""
    rng = np.random.default_rng(seed)
    sus: list[float] = []
    ref: list[float] = []
    for _ in range(N):
        sus.append(float((rng.random(M) < p_suspect).mean()))
        ref.append(float((rng.random(M) < p_ref).mean()))
    return empirical_wr_rp(sus, ref)
