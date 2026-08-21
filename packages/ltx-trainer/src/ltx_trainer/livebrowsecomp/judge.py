"""LLM-as-judge grading (Appendix B, arXiv:2605.28721)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_JUDGE_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the
precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put
the extracted answer as 'None' if there is no exact, final answer to extract from the
response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based
on [correct_answer], focusing only on if there are meaningful differences between
[correct_answer] and the extracted_final_answer.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given
above, or is within a small margin of error for numerical problems. Answer 'no'
otherwise.

confidence: The extracted confidence score between 0% and 100% from [response].
Put 100 if there is no confidence score available.
"""

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_EXTRACTED_FIELD = re.compile(
    r"extracted_final_answer:\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
_CORRECT_FIELD = re.compile(r"correct:\s*(yes|no)\b", re.IGNORECASE)


@dataclass
class JudgeResult:
    correct: bool
    extracted_answer: str
    method: str
    judge_text: str | None = None


def judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return _JUDGE_TEMPLATE.format(
        question=question.strip(),
        response=response.strip(),
        correct_answer=correct_answer.strip(),
    )


def extract_answer_from_response(response: str) -> str:
    """Prefer <answer> tags; else last non-empty line."""
    m = _ANSWER_TAG.search(response)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def parse_judge_output(judge_text: str) -> tuple[str, bool | None]:
    """Parse judge reply; returns (extracted, correct or None if unparseable)."""
    extracted = ""
    em = _EXTRACTED_FIELD.search(judge_text)
    if em:
        extracted = em.group(1).strip().strip("'\"")
        if extracted.lower() == "none":
            extracted = ""
    cm = _CORRECT_FIELD.search(judge_text)
    if cm:
        return extracted, cm.group(1).lower() == "yes"
    # Paper fallback: leading A/B
    stripped = judge_text.strip()
    if stripped.upper().startswith("A"):
        return extracted, True
    if stripped.upper().startswith("B"):
        return extracted, False
    return extracted, None


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def answers_equivalent(pred: str, gold: str) -> bool:
    """Surface-form tolerant match (abbreviations / case)."""
    p, g = _normalize(pred), _normalize(gold)
    if not g:
        return not p
    if p == g:
        return True
    if g in p or p in g:
        return len(p) <= len(g) * 2 + 4
    return False


def grade_response(
    question: str,
    response: str,
    gold: str,
    *,
    judge_output: str | None = None,
) -> JudgeResult:
    """
    Grade one model response.

    If ``judge_output`` is provided, parse the LLM judge fields; otherwise use
    deterministic equivalence on extracted final answer.
    """
    if judge_output is not None:
        extracted, ok = parse_judge_output(judge_output)
        if ok is not None:
            return JudgeResult(
                correct=ok,
                extracted_answer=extracted or extract_answer_from_response(response),
                method="llm_judge",
                judge_text=judge_output,
            )
    extracted = extract_answer_from_response(response)
    ok = answers_equivalent(extracted, gold)
    return JudgeResult(correct=ok, extracted_answer=extracted, method="rule_match")


def simulate_judge_output(question: str, response: str, gold: str) -> str:
    """Deterministic judge stand-in for offline eval (no API)."""
    extracted = extract_answer_from_response(response)
    ok = answers_equivalent(extracted, gold)
    return (
        f"extracted_final_answer: {extracted or 'None'}\n"
        f"[correct_answer]: {gold}\n"
        f"reasoning: {'match' if ok else 'mismatch'}\n"
        f"correct: {'yes' if ok else 'no'}\n"
        f"confidence: 100\n"
    )
