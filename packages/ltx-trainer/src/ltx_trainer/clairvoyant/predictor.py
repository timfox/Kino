"""Lexical length predictor — CPU stub matching XGBoost ranking behaviour (Sec. 3.3, 4)."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.clairvoyant.constants import MEDIUM_MAX_TOKENS, SHORT_MAX_TOKENS
from ltx_trainer.clairvoyant.features import extract_features


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _logits_from_features(features: dict[str, Any], *, variant: str = "sharegpt") -> list[float]:
    """
    Heuristic 3-class logits calibrated from paper ablation (prompt_token_len dominant).

    ONNX export is optional in production; this stub runs without onnxruntime.
    """
    tok_len = float(features["prompt_token_len"])
    code = float(features["has_code_keyword"])
    fmt = float(features["has_format_keyword"])
    question = float(features["ends_with_question"])
    constraint = float(features["has_length_constraint"])
    clauses = float(features["clause_count"])
    verb = str(features.get("instruction_verb") or "other")

    # Base log-odds: longer prompts → higher P(Long)
    long_score = (
        0.045 * tok_len
        + 1.2 * code
        + 0.35 * fmt
        + 0.15 * clauses
        - 0.9 * question
        - 0.5 * constraint
    )
    if verb in ("implement", "write", "generate", "describe"):
        long_score += 0.6
    if verb in ("what", "define", "summarize", "list"):
        long_score -= 0.5

    variant_bias = {"sharegpt": 0.0, "lmsys": -0.15, "oasst1": 0.1}.get(variant, 0.0)
    long_score += variant_bias

    # Three-class partition around paper boundaries
    short_logit = -long_score - 0.5
    long_logit = long_score
    medium_logit = -abs(long_score) * 0.3 + 0.2
    return [short_logit, medium_logit, long_logit]


def predict_proba(prompt: str, *, variant: str = "sharegpt") -> list[float]:
    """Return [P(Short), P(Medium), P(Long)]."""
    feats = extract_features(prompt)
    return _softmax(_logits_from_features(feats, variant=variant))


def predict_plong(prompt: str, *, variant: str = "sharegpt") -> float:
    """Priority key for SJF scheduler (ascending P(Long))."""
    return predict_proba(prompt, variant=variant)[2]


def predict_class(prompt: str, *, variant: str = "sharegpt") -> str:
    probs = predict_proba(prompt, variant=variant)
    labels = ("short", "medium", "long")
    return labels[int(max(range(3), key=lambda i: probs[i]))]


def token_class_from_length(num_tokens: int) -> str:
    if num_tokens < SHORT_MAX_TOKENS:
        return "short"
    if num_tokens < MEDIUM_MAX_TOKENS:
        return "medium"
    return "long"


def predict_record(prompt: str, *, variant: str = "sharegpt") -> dict[str, Any]:
    probs = predict_proba(prompt, variant=variant)
    feats = extract_features(prompt)
    return {
        "p_short": round(probs[0], 6),
        "p_medium": round(probs[1], 6),
        "p_long": round(probs[2], 6),
        "predicted_class": predict_class(prompt, variant=variant),
        "instruction_verb": feats["instruction_verb"],
        "prompt_token_len": feats["prompt_token_len"],
        "variant": variant,
    }
