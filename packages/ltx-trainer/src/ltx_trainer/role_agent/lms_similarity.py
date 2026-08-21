"""Longest Matching Subsequence (LMS) similarity for textual states."""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def lms_length(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence (word-level)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def lms_similarity(predicted: str, actual: str) -> float:
    """LMS score in [0, 1] per Role-Agent Eq. 4."""
    pa = tokenize(predicted)
    pb = tokenize(actual)
    if not pa and not pb:
        return 1.0
    denom = max(len(pa), len(pb), 1)
    return lms_length(pa, pb) / denom


def states_equivalent(
    a: str,
    b: str,
    *,
    threshold: float = 0.9,
) -> bool:
    return lms_similarity(a, b) >= threshold
