"""ForestTraceQA metrics: Acc, Cov, Sim, abstention (Section 4.1.2)."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def trace_coverage(pred: Sequence[str], gold: Sequence[str]) -> float:
    cp = Counter(pred)
    cg = Counter(gold)
    num = sum(min(cp[t], cg[t]) for t in cg)
    den = sum(cg.values())
    return num / den if den else 1.0


def trace_similarity(pred: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    return lcs_length(list(pred), list(gold)) / len(gold)


def answer_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def abstain_f1(tp: int, fp: int, fn: int) -> float:
    if 2 * tp + fp + fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def harmful_answer_rate(fn_abs: int, tp_abs: int) -> float:
    den = fn_abs + tp_abs
    return (fn_abs / den * 100.0) if den else 0.0
