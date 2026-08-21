"""Forward convolutive prediction (FCP) toy — Eq. (11)–(13)."""

from __future__ import annotations

import numpy as np


def fcp_weight(denominator: np.ndarray, *, xi: float, use_quantile_90: bool = False) -> np.ndarray:
    """λ in Eq. (12) or (13)."""
    if use_quantile_90:
        frame_max = np.max(np.abs(denominator) ** 2, axis=-1)
        floor = float(np.quantile(frame_max, 0.9)) if frame_max.size else 0.0
    else:
        floor = float(np.max(np.abs(denominator) ** 2))
    return xi * floor + np.abs(denominator) ** 2


def estimate_fcp_filter(
    mixture: np.ndarray,
    source_estimate: np.ndarray,
    *,
    past_taps: int,
    future_taps: int,
    xi: float = 0.01,
    use_quantile_90: bool = False,
) -> np.ndarray:
    """Closed-form FCP filter h ∈ C^{I+1+J} minimizing weighted ||Y - h^H Z̃||².

    mixture, source_estimate: shape (n_frames,) complex STFT coeffs at one frequency.
    """
    n = int(mixture.shape[0])
    taps = past_taps + 1 + future_taps
    if n <= taps:
        return np.zeros(taps, dtype=np.complex128)

    rows: list[np.ndarray] = []
    targets: list[complex] = []
    weights: list[float] = []
    for t in range(past_taps, n - future_taps):
        z_stack = []
        for lag in range(-past_taps, future_taps + 1):
            z_stack.append(source_estimate[t + lag])
        rows.append(np.asarray(z_stack, dtype=np.complex128))
        targets.append(mixture[t])
        w = fcp_weight(mixture, xi=xi, use_quantile_90=use_quantile_90)
        weights.append(float(w[t]) if w.ndim else float(w))

    z_mat = np.stack(rows, axis=0)
    y_vec = np.asarray(targets, dtype=np.complex128)
    w_vec = np.asarray(weights, dtype=np.float64)
    sqrt_w = np.sqrt(np.maximum(w_vec, 1e-12))
    zw = z_mat * sqrt_w[:, None]
    yw = y_vec * sqrt_w
    h, _, _, _ = np.linalg.lstsq(zw, yw, rcond=None)
    return h.astype(np.complex128)
