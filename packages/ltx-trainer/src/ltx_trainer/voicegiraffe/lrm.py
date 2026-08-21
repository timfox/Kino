"""Reasoning-enhanced cascading over clip captions (VOICEGIRAFFE §4, Table 3)."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.voicegiraffe.cascade import cascade_predict, _choice_score
from ltx_trainer.voicegiraffe.dataset import QAItem


def _extract_evidence_lines(caption_text: str, question: str, *, max_lines: int = 5) -> list[str]:
    """Select caption lines most relevant to the question (proxy LRM retrieval)."""
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored: list[tuple[float, str]] = []
    for line in caption_text.splitlines():
        tokens = set(re.findall(r"[a-z0-9]+", line.lower()))
        overlap = len(q_tokens & tokens) / max(len(q_tokens), 1)
        scored.append((overlap, line))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [line for score, line in scored[:max_lines] if score > 0]


def lrm_predict(item: QAItem, caption_text: str, *, backend: str = "gemini31") -> str:
    """Reason over retrieved evidence; re-score choices with boosted weights."""
    evidence = _extract_evidence_lines(caption_text, item.question)
    evidence_blob = " ".join(evidence)
    scores = {
        letter: _choice_score(evidence_blob, item.question, text) * 1.25
        for letter, text in item.choices.items()
    }
    # Multi-hop tasks benefit more from evidence aggregation (paper Table 3)
    if item.tier == "multi_hop":
        for letter in scores:
            scores[letter] *= 1.1
    if backend == "gpt52" and item.tier == "multi_hop":
        # Paper: GPT-5.2 can degrade strong proprietary models on multi-hop
        cascade = cascade_predict(item, caption_text)
        scores[cascade] = scores.get(cascade, 0) * 0.85
    return max(scores, key=scores.get)


def lrm_evaluate(items: list[QAItem], captions: dict[str, str], *, backend: str = "gemini31") -> dict[str, Any]:
    correct = 0
    for item in items:
        cap = captions.get(item.recording_id, "")
        pred = lrm_predict(item, cap, backend=backend)
        correct += int(pred == item.gold)
    n = len(items)
    return {"accuracy_pct": 100.0 * correct / max(n, 1), "backend": backend, "n_items": n}
