"""QA metrics for TaC utility reward (HotpotQA-style EM / F1)."""

from __future__ import annotations

import re
import string


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def utility_reward(prediction: str, gold: str) -> float:
    """R_utility = max(EM, F1) per paper Eq. 2."""
    em = exact_match(prediction, gold)
    f1 = f1_score(prediction, gold)
    return max(em, f1)
