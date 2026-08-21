"""Toy combined-rank filtering (Sec. 3.2) — illustrative only."""

from __future__ import annotations

import numpy as np


def mean_rank_scores(dnsmos: np.ndarray, speech_ratio: np.ndarray, wer: np.ndarray) -> np.ndarray:
    """Average per-sample rank (0 = best) across three ascending quality axes.

    Higher DNSMOS and SR are better; lower WER is better. Ranks are computed
    per-axis then averaged, matching the paper's 'combined score' intuition.
    """
    n = int(dnsmos.shape[0])
    if speech_ratio.shape[0] != n or wer.shape[0] != n:
        raise ValueError("dnsmos, speech_ratio, and wer must have the same length")

    r_dns = np.argsort(np.argsort(-dnsmos))  # high dnsmos -> low rank
    r_sr = np.argsort(np.argsort(-speech_ratio))
    r_wer = np.argsort(np.argsort(wer))  # low wer -> low rank
    return (r_dns.astype(np.float64) + r_sr.astype(np.float64) + r_wer.astype(np.float64)) / 3.0


def keep_top_percentile_by_combined_rank(
    dnsmos: np.ndarray,
    speech_ratio: np.ndarray,
    wer: np.ndarray,
    remove_bottom_pct: float = 15.0,
) -> np.ndarray:
    """Boolean mask: True if sample is kept after removing worst `remove_bottom_pct` percent.

    Mean rank is lower for better triples (high DNSMOS or SR, low WER). The worst tail
    is therefore the highest mean ranks; drop above the (100 - p) percentile.
    """
    scores = mean_rank_scores(dnsmos, speech_ratio, wer)
    thr = np.percentile(scores, 100.0 - remove_bottom_pct)
    return scores <= thr
