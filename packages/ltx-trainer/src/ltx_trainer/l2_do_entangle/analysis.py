"""CKA and stratified CER-gap analysis stubs (arXiv:2606.06065)."""

from __future__ import annotations

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA (Kornblith et al., 2019) — CPU stub."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(x.T @ x, ord="fro") ** 2
    hsic_yy = np.linalg.norm(y.T @ y, ord="fro") ** 2
    denom = hsic_xx * hsic_yy
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def stratified_cer_gaps(
    *,
    surface_gaps: dict[str, float],
    meaning_gaps: dict[str, float],
) -> list[dict[str, float | str]]:
    """Figure 2 style rows keyed by edit-distance bin."""
    rows: list[dict[str, float | str]] = []
    for ed_bin in surface_gaps:
        rows.append(
            {
                "ed_bin": ed_bin,
                "surface_gap": surface_gaps[ed_bin],
                "meaning_gap": meaning_gaps.get(ed_bin, 0.0),
            }
        )
    return rows
