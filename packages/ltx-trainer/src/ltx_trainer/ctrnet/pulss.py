"""PuLSS pseudo-label and sync-delay enumeration toy (Eqs. 25–27)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.ctrnet.fcp import estimate_fcp_filter


def enumerate_sync_delay(
    mixture: np.ndarray,
    close_talk_est: np.ndarray,
    *,
    max_delay: int,
    filter_taps: int,
) -> int:
    """Eq. (26): choose K̂ minimizing FCP residual over Ψ = {-E,…,E}."""
    best_k = 0
    best_err = float("inf")
    for k in range(-max_delay, max_delay + 1):
        shifted = np.roll(close_talk_est, k)
        if k > 0:
            shifted[:k] = 0.0
        elif k < 0:
            shifted[k:] = 0.0
        past = max(0, filter_taps - 1)
        h = estimate_fcp_filter(mixture, shifted, past_taps=past, future_taps=0, xi=0.01)
        recon = _apply_filter(mixture, shifted, h, past_taps=past, future_taps=0)
        err = float(np.mean(np.abs(mixture - recon) ** 2))
        if err < best_err:
            best_err = err
            best_k = k
    return best_k


def pseudo_label_at_reference(
    close_talk_est: np.ndarray,
    mixture: np.ndarray,
    *,
    max_delay: int = 9,
    filter_taps: int = 2,
) -> tuple[np.ndarray, int]:
    """Eq. (27): SPL_q(c) = ĥ^H Z̃(Ĉ+K̂)."""
    k_hat = enumerate_sync_delay(mixture, close_talk_est, max_delay=max_delay, filter_taps=filter_taps)
    shifted = np.roll(close_talk_est, k_hat)
    if k_hat > 0:
        shifted[:k_hat] = 0.0
    elif k_hat < 0:
        shifted[k_hat:] = 0.0
    past = max(0, filter_taps - 1)
    h = estimate_fcp_filter(mixture, shifted, past_taps=past, future_taps=0, xi=0.01)
    pl = _apply_filter(mixture, shifted, h, past_taps=past, future_taps=0)
    return pl, k_hat


def _apply_filter(
    mixture: np.ndarray,
    source: np.ndarray,
    h: np.ndarray,
    *,
    past_taps: int,
    future_taps: int,
) -> np.ndarray:
    n = mixture.shape[0]
    out = np.zeros(n, dtype=np.complex128)
    for t in range(past_taps, n - future_taps):
        z_stack = [source[t + lag] for lag in range(-past_taps, future_taps + 1)]
        out[t] = np.vdot(h, np.asarray(z_stack, dtype=np.complex128))
    return out
