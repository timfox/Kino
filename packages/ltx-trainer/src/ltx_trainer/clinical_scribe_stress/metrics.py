"""Cause-side and result-side metrics (Eq. 1–7, §3.4)."""

from __future__ import annotations


def wer(substitutions: int, deletions: int, insertions: int, ref_words: int) -> float:
    """Micro-averaged WER in percent (Eq. 1)."""
    if ref_words == 0:
        return 0.0
    return 100.0 * (substitutions + deletions + insertions) / ref_words


def ins_rate(insertions: int, ref_words: int) -> float:
    """Insertion rate in percent (Eq. 2)."""
    if ref_words == 0:
        return 0.0
    return 100.0 * insertions / ref_words


def neg_err_rate(negation_mismatches: int, n_dialogues: int) -> float:
    """Negation error rate in percent (Eq. 3)."""
    if n_dialogues == 0:
        return 0.0
    return 100.0 * negation_mismatches / n_dialogues


def triage_match(clean_labels: list[int], noisy_labels: list[int]) -> float:
    """Paired triage invariance fraction (Eq. 4)."""
    if not clean_labels:
        return 1.0
    matches = sum(1 for c, n in zip(clean_labels, noisy_labels, strict=True) if c == n)
    return matches / len(clean_labels)


def under_triage_rate(new_under: int, n_dialogues: int) -> float:
    """Noise-induced under-triage rate in percent (Eq. 5)."""
    if n_dialogues == 0:
        return 0.0
    return 100.0 * new_under / n_dialogues


def scer_rate(new_critical_fp: int, n_dialogues: int) -> float:
    """Safety-critical error rate in percent (Eq. 6)."""
    if n_dialogues == 0:
        return 0.0
    return 100.0 * new_critical_fp / n_dialogues


def err_prop(delta_claim_errors: int, delta_asr_errors: int) -> float:
    """Error propagation: new claim errors per 100 ΔASR errors (Eq. 7)."""
    if delta_asr_errors <= 0:
        return 0.0
    return 100.0 * delta_claim_errors / delta_asr_errors


def unsafe_rate(scores: list[int], *, threshold: int = 2) -> float:
    """Fraction of notes scored ≤ threshold on 1–5 rubric (§3.4.3)."""
    if not scores:
        return 0.0
    return 100.0 * sum(1 for s in scores if s <= threshold) / len(scores)
