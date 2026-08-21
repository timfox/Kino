"""VTM complexity and encoding-time ratios (Eq. 1–2)."""

from __future__ import annotations


def complexity_reduction_pct(
    reference_per_qp: dict[int, float],
    candidate_per_qp: dict[int, float],
    *,
    qps: tuple[int, ...] = (22, 27, 32, 37),
) -> float:
    """Eq. (1) as written: average over QPs of 100·(C_ref − C_cand) / C_ref."""
    terms: list[float] = []
    for qp in qps:
        ref = reference_per_qp[qp]
        cand = candidate_per_qp[qp]
        if ref <= 0:
            raise ValueError(f"reference count must be positive at QP {qp}")
        terms.append(100.0 * (ref - cand) / ref)
    return sum(terms) / len(terms)


def relative_complexity_pct(
    reference_per_qp: dict[int, float],
    candidate_per_qp: dict[int, float],
    *,
    qps: tuple[int, ...] = (22, 27, 32, 37),
) -> float:
    """
    Table 1 style: 100·C_cand / C_ref (CU or pixel counts vs VTM-18.0).

    Values near 100 mean similar RDO exploration; below 100 means fewer reconstructions.
    """
    terms: list[float] = []
    for qp in qps:
        ref = reference_per_qp[qp]
        cand = candidate_per_qp[qp]
        if ref <= 0:
            raise ValueError(f"reference count must be positive at QP {qp}")
        terms.append(100.0 * cand / ref)
    return sum(terms) / len(terms)


def complexity_ratio(
    reference_per_qp: dict[int, float],
    candidate_per_qp: dict[int, float],
    *,
    qps: tuple[int, ...] = (22, 27, 32, 37),
) -> float:
    """Alias for Table 1 relative CU/pixel ratio (% of reference)."""
    return relative_complexity_pct(reference_per_qp, candidate_per_qp, qps=qps)


def encoding_time_ratio(
    reference_time_per_qp: dict[int, float],
    candidate_time_per_qp: dict[int, float],
    *,
    qps: tuple[int, ...] = (22, 27, 32, 37),
) -> float:
    """
    Eq. (2) / Table 1 ET column: 100·T_cand / T_ref.

    ET below 100% means faster encoding than reference (e.g. VTM-23.11 at 93.5%).
    """
    return relative_complexity_pct(reference_time_per_qp, candidate_time_per_qp, qps=qps)
