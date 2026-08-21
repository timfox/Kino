"""Detection and recovery metrics — Appendix A.5."""

from __future__ import annotations


def recall_fpr(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float]:
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return float(recall * 100), float(fpr * 100)


def wer(substitutions: int, deletions: int, insertions: int, n_words: int) -> float:
    if n_words == 0:
        return 0.0
    return 100.0 * (substitutions + deletions + insertions) / n_words


def werr(wer_initial: float, wer_final: float) -> float:
    if wer_initial <= 0:
        return 0.0
    return 100.0 * (wer_initial - wer_final) / wer_initial
