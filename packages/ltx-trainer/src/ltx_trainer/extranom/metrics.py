"""Caption similarity and VAD AUC helpers (Tables 3–4)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def unigram_f1(pred: str, ref: str) -> float:
    """Token F1 proxy for caption overlap (smoke; not full BLEU)."""
    p = Counter(_tokenize(pred))
    r = Counter(_tokenize(ref))
    if not p or not r:
        return 0.0
    overlap = sum((p & r).values())
    prec = overlap / sum(p.values())
    rec = overlap / sum(r.values())
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l_recall(pred: str, ref: str) -> float:
    """Longest common subsequence recall (ROUGE-L style)."""
    p = _tokenize(pred)
    r = _tokenize(ref)
    if not r:
        return 0.0
    m, n = len(p), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / n


def caption_metrics(pred: str, ref: str) -> dict[str, float]:
    """Lightweight caption scores for unit tests (Table 3 uses full BLEU/BERT/CIDEr)."""
    f1 = unigram_f1(pred, ref)
    rl = rouge_l_recall(pred, ref)
    return {
        "token_f1": f1,
        "rouge_l_recall": rl,
        "bleu_proxy": f1 * 0.9,
        "meteor_proxy": (f1 + rl) / 2.0,
    }


def mean_caption_metrics(preds: Sequence[str], refs: Sequence[str]) -> dict[str, float]:
    if not preds or len(preds) != len(refs):
        return {k: 0.0 for k in ("token_f1", "rouge_l_recall", "bleu_proxy", "meteor_proxy")}
    keys = ("token_f1", "rouge_l_recall", "bleu_proxy", "meteor_proxy")
    sums = {k: 0.0 for k in keys}
    for p, r in zip(preds, refs, strict=True):
        m = caption_metrics(p, r)
        for k in keys:
            sums[k] += m[k]
    n = len(preds)
    return {k: sums[k] / n for k in keys}
