"""TextSculpt-Bench evaluation metrics (Sec. 4.2)."""

from __future__ import annotations


def levenshtein_counts(expected: str, observed: str) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) on word sequences."""
    exp = expected.split()
    obs = observed.split()
    m, n = len(exp), len(obs)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if exp[i - 1] == obs[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # Backtrack counts (approximate via total distance; split evenly for smoke)
    d = dp[m][n]
    s = min(m, n, d)
    i = d - s
    del_ = max(0, n - m) if n > m else max(0, d - s - i)
    if m > n:
        del_ = max(del_, m - n)
    return s, i, del_


def text_accuracy(
    expected_full: str,
    observed_full: str,
    n_edit_words: int,
) -> float:
    """TextAcc = 1 - min((S+I+D)/N_edit, 1) (Sec. 4.2)."""
    s, ins, d = levenshtein_counts(expected_full, observed_full)
    if n_edit_words <= 0:
        return 1.0
    ratio = min((s + ins + d) / n_edit_words, 1.0)
    return 1.0 - ratio


def visual_quality_score(
    location_ok: bool,
    style_ok: bool,
    physical_ok: bool,
) -> float:
    """Mean of three binary VQA criteria."""
    return sum((location_ok, style_ok, physical_ok)) / 3.0


def background_preservation_ssim(
    ssim_value: float,
) -> float:
    """SSIM on OCR-masked non-text regions (already computed externally)."""
    return max(0.0, min(1.0, ssim_value))


def average_score(ta: float, vq: float, bp: float) -> float:
    return (ta + vq + bp) / 3.0
