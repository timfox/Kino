"""Participant-level classification metrics."""

from __future__ import annotations

import numpy as np


def macro_f1_from_confusion(tn: int, fp: int, fn: int, tp: int) -> float:
    prec_pos = tp / max(tp + fp, 1)
    rec_pos = tp / max(tp + fn, 1)
    prec_neg = tn / max(tn + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    f1_pos = 2 * prec_pos * rec_pos / max(prec_pos + rec_pos, 1e-12)
    f1_neg = 2 * prec_neg * rec_neg / max(prec_neg + rec_neg, 1e-12)
    return float(0.5 * (f1_pos + f1_neg))


def paired_shift(heavy_probs: np.ndarray, neutral_probs: np.ndarray) -> float:
    """Mean p(heavy) - p(neutral) per participant."""
    return float(np.mean(heavy_probs - neutral_probs))
