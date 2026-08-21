"""Mixed Error Rate (MER) helpers."""

from __future__ import annotations


def mer(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """Word/phoneme-level mixed error rate in [0, 1]."""
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    # Levenshtein at token level
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def average_mer(pair_mers: dict[str, float], pairs: tuple[str, ...]) -> float:
    vals = [pair_mers[p] for p in pairs]
    return round(sum(vals) / len(vals), 2)
