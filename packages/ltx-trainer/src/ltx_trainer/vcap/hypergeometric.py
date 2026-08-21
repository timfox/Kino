"""Hypergeometric recall and collision factors (Eq. 2–3, Sec. 2.3)."""

from __future__ import annotations


def p_recall(c: int, m: int, N: int) -> float:
    """Precall(c | m, N) = ∏_{i=0}^{m-1} (c-i)/(N-i). Completeness axis."""
    if m <= 0:
        return 1.0
    if c < m or N <= 0 or c > N:
        return 0.0
    prod = 1.0
    for i in range(m):
        denom = N - i
        if denom <= 0:
            return 0.0
        prod *= (c - i) / denom
    return max(0.0, min(1.0, prod))


def p_collision(n_err: int, m: int, N: int) -> float:
    """Pcoll(n-c | m, N) = ∏_{j=0}^{n_err-1} (N-m-j)/(N-j). Correctness axis."""
    if n_err <= 0:
        return 1.0
    if N <= m:
        return 0.0
    prod = 1.0
    for j in range(n_err):
        denom = N - j
        if denom <= 0:
            return 0.0
        prod *= (N - m - j) / denom
    return max(0.0, min(1.0, prod))


def fact_counts_from_scores(
    *,
    c: int,
    n_total: int,
    m_witness: int,
    N: int,
) -> dict[str, float]:
    """Map discrete fact counts to hypergeometric factors."""
    n_err = max(0, n_total - c)
    return {
        "p_recall": p_recall(c, m_witness, N),
        "p_collision": p_collision(n_err, m_witness, N),
        "c": float(c),
        "n_err": float(n_err),
        "m": float(m_witness),
        "N": float(N),
    }
