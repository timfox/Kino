"""Toy GOP and GOP-feature extraction utilities.

The paper uses Kaldi phone posteriors to compute phoneme-level GOP and 2K GOP features.
This stub only implements the core math on arrays.
"""

from __future__ import annotations

import numpy as np


def gop_from_log_posteriors(log_posteriors: np.ndarray, target_phone_id: int) -> float:
    """Compute GOP(p) as average LPP over an aligned segment.

    Args:
        log_posteriors: shape (T, K), log P_t(phone | O) for frames t in [ts, te].
        target_phone_id: integer in [0, K).
    """
    if log_posteriors.ndim != 2:
        raise ValueError("log_posteriors must be 2D (T,K)")
    if log_posteriors.shape[0] == 0:
        return 0.0
    k = int(target_phone_id)
    if k < 0 or k >= log_posteriors.shape[1]:
        raise ValueError("target_phone_id out of range")
    return float(log_posteriors[:, k].mean())


def gop_feature_vector(log_posteriors: np.ndarray, target_phone_id: int) -> np.ndarray:
    """2K GOP features for one phoneme segment (toy).

    First K: LPP for all phones (averaged over aligned frames).
    Last K: LPP(target) - LPP(j) for each phone j.
    """
    if log_posteriors.ndim != 2:
        raise ValueError("log_posteriors must be 2D (T,K)")
    if log_posteriors.shape[0] == 0:
        k = log_posteriors.shape[1]
        return np.zeros((2 * k,), dtype=np.float64)

    lpp = log_posteriors.mean(axis=0).astype(np.float64, copy=False)
    k = lpp.shape[0]
    t = int(target_phone_id)
    if t < 0 or t >= k:
        raise ValueError("target_phone_id out of range")
    diffs = lpp[t] - lpp
    return np.concatenate([lpp, diffs], axis=0)


def batch_gop_features(
    segments: list[tuple[np.ndarray, int]],
) -> np.ndarray:
    """Stack multiple GOP feature vectors to shape (N, 2K)."""
    if not segments:
        return np.zeros((0, 0), dtype=np.float64)
    feats = [gop_feature_vector(lp, pid) for lp, pid in segments]
    return np.stack(feats, axis=0)
