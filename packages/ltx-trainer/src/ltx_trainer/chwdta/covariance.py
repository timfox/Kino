"""Covariance sparsification diagnostics (Eq. 7–9, 12)."""

from __future__ import annotations

import numpy as np


def roff_block_ratio(cov: np.ndarray) -> float:
    """Cross-branch coupling ratio (Eq. 8) for 2×2 block layout."""
    c = cov.shape[0]
    half = c // 2
    ll = cov[:half, :half]
    lh = cov[:half, half:]
    hl = cov[half:, :half]
    hh = cov[half:, half:]
    num = np.linalg.norm(lh, "fro") + np.linalg.norm(hl, "fro")
    den = np.linalg.norm(ll, "fro") + np.linalg.norm(hh, "fro")
    return float(num / (den + 1e-8))


def rent_ratio(cov: np.ndarray) -> float:
    """Entropy mismatch ratio (Eq. 12) smoke on slice blocks."""
    s = cov.shape[0]
    off_cross = 0.0
    off_within = 0.0
    for i in range(s):
        for j in range(s):
            block = cov[i * s // 8 : (i + 1) * s // 8, j * s // 8 : (j + 1) * s // 8] if s >= 8 else cov
            if i != j:
                off_cross += np.linalg.norm(block, "fro")
            else:
                d = block.copy()
                np.fill_diagonal(d, 0)
                off_within += np.linalg.norm(d, "fro")
    total = np.linalg.norm(cov, "fro")
    return float((off_cross + off_within) / (total + 1e-8))
