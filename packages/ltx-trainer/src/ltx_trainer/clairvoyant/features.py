"""19 lightweight lexical features for output-length prediction (Sec. 3.2)."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.clairvoyant.constants import (
    CLAUSE_MARKERS,
    CODE_KEYWORDS,
    FORMAT_KEYWORDS,
    INSTRUCTION_VERBS,
    LENGTH_CONSTRAINT_KEYWORDS,
)


def _first_token(prompt: str) -> str:
    stripped = prompt.lstrip()
    if not stripped:
        return ""
    return re.split(r"[\s:,]+", stripped, maxsplit=1)[0].lower().strip("\"'`")


def extract_instruction_verb(prompt: str) -> str:
    """Map leading token to one of 13 verb categories."""
    tok = _first_token(prompt)
    if tok in INSTRUCTION_VERBS:
        return tok
    # common variants
    aliases = {
        "summarise": "summarize",
        "create": "generate",
        "build": "implement",
        "tell": "explain",
        "show": "describe",
    }
    if tok in aliases:
        return aliases[tok]
    return "other"


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def clause_count(prompt: str) -> int:
    lower = f" {prompt.lower()} "
    return sum(lower.count(marker) for marker in CLAUSE_MARKERS)


def extract_features(prompt: str) -> dict[str, Any]:
    """Return numeric + verb features and flat vector for predictor."""
    verb = extract_instruction_verb(prompt)
    numeric = {
        "prompt_token_len": max(1, len(prompt) // 4),
        "has_code_keyword": int(_contains_keyword(prompt, CODE_KEYWORDS)),
        "has_length_constraint": int(_contains_keyword(prompt, LENGTH_CONSTRAINT_KEYWORDS)),
        "ends_with_question": int(prompt.rstrip().endswith("?")),
        "has_format_keyword": int(_contains_keyword(prompt, FORMAT_KEYWORDS)),
        "clause_count": clause_count(prompt),
    }
    verb_one_hot = {f"verb_{v}": int(verb == v) for v in INSTRUCTION_VERBS}
    features = {**numeric, "instruction_verb": verb, **verb_one_hot}
    vector = [
        numeric["prompt_token_len"],
        numeric["has_code_keyword"],
        numeric["has_length_constraint"],
        numeric["ends_with_question"],
        numeric["has_format_keyword"],
        numeric["clause_count"],
    ] + [verb_one_hot[f"verb_{v}"] for v in INSTRUCTION_VERBS]
    features["vector"] = vector
    return features


def feature_names() -> list[str]:
    numeric = [
        "prompt_token_len",
        "has_code_keyword",
        "has_length_constraint",
        "ends_with_question",
        "has_format_keyword",
        "clause_count",
    ]
    return numeric + [f"verb_{v}" for v in INSTRUCTION_VERBS]
