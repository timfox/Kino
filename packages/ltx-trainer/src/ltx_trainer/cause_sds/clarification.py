"""K-round clarification loop — §3.5, Figure 2."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cause_sds.detectors import ErrorCause


def strategy_for_cause(cause: ErrorCause) -> str:
    """Targeted recovery strategy per failure mode."""
    strategies = {
        ErrorCause.PERCEPTION: "repeat_or_quieter_room",
        ErrorCause.COMPREHENSION: "clarify_or_spell",
        ErrorCause.DELETION: "repeat_missing_segment",
    }
    return strategies[cause]


def clarification_query(cause: ErrorCause, token_span: str) -> str:
    if cause == ErrorCause.PERCEPTION:
        return (
            f"Could you repeat '{token_span}' (maybe closer to the mic or in a quieter spot)?"
        )
    if cause == ErrorCause.COMPREHENSION:
        return f"Did you mean a different word for '{token_span}'? Can you spell it?"
    return f"I didn't catch '{token_span}' — could you say that part again?"


def run_clarification_round(
    transcript: str,
    unresolved_spans: list[tuple[str, ErrorCause]],
    user_response: str,
) -> dict[str, Any]:
    """Single round: query → user text → updated transcript (toy merge)."""
    if not unresolved_spans:
        return {"transcript": transcript, "resolved": True, "queries": []}
    span, cause = unresolved_spans[0]
    query = clarification_query(cause, span)
    updated = transcript.replace(span, user_response.strip(), 1) if span in transcript else transcript
    return {
        "transcript": updated,
        "resolved": span not in updated or user_response.strip() != "",
        "queries": [query],
        "strategy": strategy_for_cause(cause),
    }


def k_round_clarification(
    initial_transcript: str,
    spans: list[tuple[str, ErrorCause]],
    user_responses: list[str],
    k_max: int = 3,
) -> dict[str, Any]:
    transcript = initial_transcript
    history: list[dict[str, Any]] = []
    remaining = list(spans)
    for i in range(min(k_max, len(user_responses) + 1)):
        if not remaining:
            break
        resp = user_responses[i] if i < len(user_responses) else ""
        step = run_clarification_round(transcript, remaining, resp)
        history.append(step)
        transcript = step["transcript"]
        if step.get("resolved"):
            remaining = remaining[1:]
    return {"U_final": transcript, "rounds": history, "remaining": len(remaining)}
