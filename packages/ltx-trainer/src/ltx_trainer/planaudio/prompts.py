"""Prompt templates and LTX caption hints for compositional audio."""

from __future__ import annotations

PLANAUDIO_CAPTION_HINT = """\
Describe audio for unified speech-and-sound generation: separate quoted speech (exact words) \
from non-speech events (music, ambience, effects). Note temporal order (e.g. before/after/then) \
and interactions (applause after speech, voice over music)."""

LTX_AV_PROMPT_SUFFIX = """\
PlanAudio-style free-form target: one natural-language line may specify both environmental \
sound and quoted dialogue with clear timing. Prefer explicit event order over vague mixing."""


def rewrite_prompt_stub(free_form: str, *, scenario: str) -> str:
    """VoiceLDM-style multi-input rewrite (Table 1) for pipeline baselines — not used by PlanAudio."""
    text = free_form.strip()
    if scenario == "speech":
        speech = _extract_quoted(text) or text
        return f"[Sound]: clean speech for an audiobook.\n[Speech]: {speech}"
    if scenario == "sound":
        return f"[Sound]: {text}\n[Speech]:"
    speech = _extract_quoted(text) or ""
    sound = _strip_quotes(text) or text
    return f"[Sound]: {sound}\n[Speech]: {speech}"


def _extract_quoted(text: str) -> str:
    for q in ('"', "'"):
        if q in text:
            start = text.find(q) + 1
            end = text.find(q, start)
            if end > start:
                return text[start:end]
    return ""


def _strip_quotes(text: str) -> str:
    out = text
    for q in ('"', "'"):
        while q in out:
            start = out.find(q)
            end = out.find(q, start + 1)
            if end < 0:
                break
            out = (out[:start] + out[end + 1 :]).strip()
    return out.strip(" ,:;")
