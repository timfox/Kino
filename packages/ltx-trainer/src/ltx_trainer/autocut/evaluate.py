"""GPT-4o evaluation prompts and local proxies (Sec. 4.2, Supp. 7.3)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.autocut.metrics import (
    script_quality_heuristic,
    visual_script_correlation_score,
)

VSC_PROMPT = """You are evaluating Visual–Script Correlation for an advertisement clip.
Given one video frame description and one script line, score semantic alignment:
0 = mismatch, 1 = partial relevance, 2 = strong alignment.
Return only the integer score."""

SQ_PROMPT = """You are evaluating Script Quality for an advertisement on a 100-point rubric:
(1) Basic Quality 30pts — factual correctness, clarity, safety;
(2) Expression 40pts — natural language, engagement, selling points;
(3) Length/Rhythm 30pts — line length consistency and pacing.
Return JSON with category scores and total."""


def evaluate_vsc(frame_caption: str, script_line: str, *, use_llm: bool = False) -> int:
    if use_llm:
        raise NotImplementedError("Wire OpenAI GPT-4o for production VSC scoring")
    return visual_script_correlation_score(frame_caption, script_line)


def evaluate_sq(
    generated: str,
    reference: str,
    *,
    product_keywords: list[str] | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    if use_llm:
        raise NotImplementedError("Wire OpenAI GPT-4o for production SQ scoring")
    score = script_quality_heuristic(generated, reference, product_keywords=product_keywords or [])
    return {
        "total": score,
        "basic_quality": min(30.0, score * 0.3),
        "expression": min(40.0, score * 0.4),
        "length_rhythm": min(30.0, score * 0.3),
        "mode": "heuristic_proxy",
    }
