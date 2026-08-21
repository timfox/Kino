"""DG-WGPO reward components (Eq. 1–8)."""

from __future__ import annotations

import re

ATOMIC_PHENOMENA: tuple[str, ...] = (
    "noise",
    "far-field",
    "obstructed",
    "echo&reverb",
    "recording",
    "electronic_distortion",
    "transmission_dropout",
)


def token_edit_similarity(hyp_token: str, ref_token: str) -> float:
    """Eq. (4) character-level similarity in [0, 1]."""
    if not hyp_token and not ref_token:
        return 1.0
    if not hyp_token or not ref_token:
        return 0.0
    # Levenshtein distance at char level
    a, b = hyp_token, ref_token
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[m][n]
    denom = max(len(a), len(b))
    return 1.0 - dist / denom if denom else 1.0


def word_wer(hypothesis: str, reference: str) -> float:
    """Word-error rate in [0, 1]."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    m, n = len(hyp_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if hyp_words[i - 1] == ref_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / len(ref_words)


def wer_reward(hypothesis: str, reference: str) -> float:
    """Eq. (1): R_wer = 1 − WER."""
    return 1.0 - word_wer(hypothesis, reference)


def repetition_gate(hypothesis: str, *, ngram: int = 4, max_repeats: int = 3) -> float:
    """Eq. (2): zero if repeated n-grams exceed threshold."""
    words = hypothesis.lower().split()
    if len(words) < ngram * max_repeats:
        return 1.0
    for i in range(len(words) - ngram * max_repeats + 1):
        gram = tuple(words[i : i + ngram])
        count = 1
        j = i + ngram
        while j + ngram <= len(words):
            if tuple(words[j : j + ngram]) == gram:
                count += 1
                j += ngram
            else:
                break
        if count >= max_repeats:
            return 0.0
    return 1.0


def static_reward(hypothesis: str, reference: str) -> float:
    """Eq. (3): R_static = R_rep · R_wer."""
    return repetition_gate(hypothesis) * wer_reward(hypothesis, reference)


def token_level_fine_reward(hypothesis: str, reference: str, *, soft_alpha: float = 0.4) -> float:
    """Eq. (5): R_fine with soft/hard substitution discount."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            if cost == 0:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "eq"
            elif dp[i - 1][j] + 1 < dp[i][j - 1] + 1 and dp[i - 1][j] + 1 <= dp[i - 1][j - 1] + 1:
                dp[i][j] = dp[i - 1][j] + 1
                back[i][j] = "del"
            elif dp[i][j - 1] + 1 < dp[i - 1][j - 1] + 1:
                dp[i][j] = dp[i][j - 1] + 1
                back[i][j] = "ins"
            else:
                dp[i][j] = dp[i - 1][j - 1] + 1
                back[i][j] = "sub"
    n_c = n_hard = n_soft = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            n_hard += 1
            j -= 1
            continue
        if j == 0:
            n_hard += 1
            i -= 1
            continue
        op = back[i][j]
        if op == "eq":
            n_c += 1
            i -= 1
            j -= 1
        elif op == "sub":
            sim = token_edit_similarity(hyp[i - 1], ref[j - 1])
            if sim >= 0.5:
                n_soft += 1
            else:
                n_hard += 1
            i -= 1
            j -= 1
        else:
            n_hard += 1
            if op == "del":
                i -= 1
            else:
                j -= 1
    eps = 1e-8
    return n_c / (n_c + n_hard + soft_alpha * n_soft + eps)


def lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def sentence_level_struct_reward(hypothesis: str, reference: str) -> float:
    """Eq. (6): LCS backbone + length penalty."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0
    lcs = lcs_length(hyp, ref)
    lcs_term = 0.5 * lcs / len(ref)
    len_term = 0.5 * max(0.0, 1.0 - abs(len(hyp) - len(ref)) / len(ref))
    return lcs_term + len_term


def dynamic_reward(
    hypothesis: str,
    reference: str,
    *,
    tau: float = 0.3,
    soft_alpha: float = 0.4,
) -> float:
    """Eq. (7): WER-gated fusion of R_fine and R_struc."""
    wer = word_wer(hypothesis, reference)
    r_fine = token_level_fine_reward(hypothesis, reference, soft_alpha=soft_alpha)
    r_struc = sentence_level_struct_reward(hypothesis, reference)
    if wer < tau:
        return 0.75 * r_fine + 0.25 * r_struc
    return 0.25 * r_fine + 0.75 * r_struc


def dg_wgpo_reward(
    hypothesis: str,
    reference: str,
    *,
    tau: float = 0.3,
    soft_alpha: float = 0.4,
    alpha_dyn: float = 0.6,
) -> dict[str, float]:
    """Eq. (8): R = (1−α_dyn) R_static + α_dyn R_dynamic."""
    r_static = static_reward(hypothesis, reference)
    r_dyn = dynamic_reward(hypothesis, reference, tau=tau, soft_alpha=soft_alpha)
    total = (1.0 - alpha_dyn) * r_static + alpha_dyn * r_dyn
    return {
        "total": total,
        "static": r_static,
        "dynamic": r_dyn,
        "wer": word_wer(hypothesis, reference),
        "wer_reward": wer_reward(hypothesis, reference),
        "repetition_gate": repetition_gate(hypothesis),
    }
