"""LOC normalization and feasibility helpers (§III-C)."""

from __future__ import annotations

import re
from typing import Any


def normalize_loc(source: str) -> int:
    """Remove C/C++ comments and blank lines; preserve string literals."""
    # Strip block comments
    text = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    # Strip line comments (naive: not inside strings)
    lines: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"//.*$", "", line).strip()
        if stripped:
            lines.append(stripped)
    return len(lines)


def relative_error(x: float, xref: float, *, floor: float = 1e-12) -> float:
    denom = max(abs(xref), floor)
    return abs(x - xref) / denom


def passes_validation(
    rel_err: float,
    *,
    tol: float,
    abs_err: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    if rel_err > tol:
        return False
    if abs_tol is not None and abs_err is not None and abs_err > abs_tol:
        return False
    return True


def feasibility_summary(
    successes: int,
    trials: int,
    *,
    median_perf: float | None = None,
) -> dict[str, Any]:
    rate = round(100.0 * successes / trials, 1) if trials else 0.0
    out: dict[str, Any] = {"successes": successes, "trials": trials, "rate_pct": rate}
    if median_perf is not None:
        out["median_perf_successful"] = median_perf
    return out


def end_to_end_expected_perf(success_rate: float, median_successful: float) -> float:
    """Rough E[perf] = P(success) × E[perf | success] using conditional median."""
    return round(success_rate * median_successful, 3)
