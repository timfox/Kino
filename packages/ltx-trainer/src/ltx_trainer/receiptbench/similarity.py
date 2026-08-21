"""String and list similarity for evaluation (Sec. 3.4, Eq. 1)."""

from __future__ import annotations

import re
from typing import Any


def normalize_string(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    a, b = normalize_string(a), normalize_string(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    dist = prev[-1]
    return 1.0 - dist / max(la, lb)


def token_sort_ratio(a: str, b: str) -> float:
    ta = sorted(normalize_string(a).split())
    tb = sorted(normalize_string(b).split())
    return levenshtein_ratio(" ".join(ta), " ".join(tb))


def lcs_ratio(a: str, b: str) -> float:
    a, b = normalize_string(a), normalize_string(b)
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[la][lb]
    return (2.0 * lcs) / (la + lb)


def ngram_cosine(a: str, b: str, n: int = 3) -> float:
    """Lightweight semantic proxy via character n-gram cosine."""
    a, b = normalize_string(a), normalize_string(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def vec(s: str) -> dict[str, int]:
        if len(s) < n:
            s = s + (" " * (n - len(s)))
        grams: dict[str, int] = {}
        for i in range(len(s) - n + 1):
            g = s[i : i + n]
            grams[g] = grams.get(g, 0) + 1
        return grams

    va, vb = vec(a), vec(b)
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = sum(v * v for v in va.values()) ** 0.5
    nb = sum(v * v for v in vb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def composite_similarity(
    a: str,
    b: str,
    *,
    alpha: float = 0.3,
    beta: float = 0.2,
    gamma: float = 0.1,
    delta: float = 0.4,
) -> float:
    """S_ij = α·Slev + β·Ssort + γ·Slcs + δ·Ssem (Eq. 1)."""
    return (
        alpha * levenshtein_ratio(a, b)
        + beta * token_sort_ratio(a, b)
        + gamma * lcs_ratio(a, b)
        + delta * ngram_cosine(a, b)
    )


def parse_amount(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None
