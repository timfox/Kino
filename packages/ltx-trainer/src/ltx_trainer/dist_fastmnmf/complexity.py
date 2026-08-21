"""Table I computational complexity summaries (§III-B)."""

from __future__ import annotations

from typing import Any


def complexity_table_rows() -> list[dict[str, str]]:
    """Table I — per iteration, per frequency bin (as in paper)."""
    return [
        {
            "component": "W_i",
            "fastmnmf_all": "O(M^4 + J M^3)",
            "distributed": "O(Σ_l M^(l)^4 + J Σ_l M^(l)^3)",
        },
        {
            "component": "t_ikn, v_kjn, Λ_in",
            "fastmnmf_all": "O(J M^2 + J N (K + M))",
            "distributed": "O(J Σ_l M^(l)^2 + J N (K + M))",
        },
        {
            "component": "Total",
            "fastmnmf_all": "O(M^4 + J M^3 + J N (K + M))",
            "distributed": "O(Σ_l M^(l)^4 + J Σ_l M^(l)^3 + J N (K + M))",
        },
    ]


def asymptotic_speedup_equal_subarrays(*, l_subarrays: int) -> dict[str, float]:
    """When all subarrays have M/L mics each, inversion cost scales ~1/L^3 vs all-array."""
    if l_subarrays < 1:
        return {"inversion_factor": 1.0, "matmul_factor": 1.0}
    return {
        "inversion_factor": float(l_subarrays**3),
        "matmul_factor": float(l_subarrays**2),
    }


def relative_runtime_vs_all(m_dist: float, m_all: float) -> float:
    """Table II ratio: distributed / FastMNMF (all subarrays)."""
    if m_all <= 0.0:
        return 0.0
    return float(m_dist / m_all)
