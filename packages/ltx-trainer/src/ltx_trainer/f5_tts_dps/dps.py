"""Dual-scoring prompt selection (LALM audio + LLM text alignment)."""

from __future__ import annotations

import re
from typing import Any

AUDIO_SCORING_PROMPT_HEAD = (
    "You are an audio expressiveness expert. Evaluate prompt suitability for expressive TTS. "
    "Score 0-10 on emotional richness, voice expressiveness, and prompt suitability."
)

TEXT_SCORING_PROMPT_HEAD = (
    "You are a speech synthesis expert. Select the best reference text for prosodic and emotional "
    "alignment with the target text. Output the selected reference in <answer></answer>."
)


def audio_expressiveness_score_stub(
    transcript: str,
    *,
    min_words: int = 5,
) -> float:
    """Toy LALM audio score proxy from transcript cues (not real audio inference)."""
    text = transcript.strip().lower()
    if len(text.split()) < min_words:
        return 3.0
    score = 5.0
    expressive = ("!", "?", "very", "really", "love", "hate", "excited", "sad", "happy")
    if any(tok in text for tok in expressive):
        score += 2.0
    if len(text) > 80:
        score += 1.0
    return min(10.0, score)


def text_alignment_pick_stub(target: str, references: list[str]) -> dict[str, Any]:
    """Toy LLM reference picker: longest token overlap with target."""
    target_tokens = set(re.findall(r"[a-z0-9']+", target.lower()))
    best = references[0] if references else ""
    best_overlap = -1
    for ref in references:
        ref_tokens = set(re.findall(r"[a-z0-9']+", ref.lower()))
        overlap = len(target_tokens & ref_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = ref
    return {"selected_reference": best, "overlap_tokens": best_overlap}


def dual_score_filter(
    candidates: list[dict[str, Any]],
    *,
    audio_min: float = 7.0,
) -> list[dict[str, Any]]:
    """Keep candidates passing audio threshold, then pick best text alignment."""
    passed: list[dict[str, Any]] = []
    for c in candidates:
        audio = c.get("audio_score")
        if audio is None:
            audio = audio_expressiveness_score_stub(str(c.get("transcript", "")))
        if audio < audio_min:
            continue
        target = str(c.get("target_text", ""))
        refs = list(c.get("reference_texts", []))
        if refs:
            pick = text_alignment_pick_stub(target, refs)
            c = {**c, "audio_score": audio, **pick}
        else:
            c = {**c, "audio_score": audio}
        passed.append(c)
    return passed
