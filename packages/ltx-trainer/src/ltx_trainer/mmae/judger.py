"""Qwen3-Omni judger prompts and mock responses (Appendix C)."""

from __future__ import annotations

import json
import random
from ltx_trainer.mmae.audio_refs import parse_audio_refs as parse_audio_ref_tokens

JUDGER_SYSTEM_PROMPT = """You are an audio analysis assistant.
Your task is to answer user questions strictly based on the factual content of the provided audio clips. Carefully
listen to and analyze the audio before answering.
Audio reference notation:
- A bare reference like <audio output> or <audio input1> refers to the full duration of that audio clip.
- A reference with [start:end] denotes a time slice of that clip (e.g. <audio output[0.0s:3s]> means the segment
from 0.0s to 3.0s).
- An omitted start means the beginning of the clip (e.g. <audio input1[:2.0s]> means from the beginning up
to 2.0s).
- An omitted end means the end of the clip (e.g. <audio output[1.5s:]> means from 1.5s to the end).
- A negative value counts backward from the end (e.g. <audio input2[-2.5s:]> means the last 2.5 seconds of
that clip).
- The slicing semantics follow Python-style [start:end] conventions.
Guidelines:
- Base your answers only on information that is clearly present or can be directly inferred from the audio.
- Do NOT make up details that are not supported by the audio.
- If the audio does not contain enough information to answer the question, say so explicitly.
- When relevant, reference specific parts of the audio (e.g., events, sounds, speech content, timing).
- Be concise, clear, and accurate.
- If the audio contains speech, you may transcribe or summarize relevant portions to support your answer.
- If the audio is noisy, ambiguous, or unclear, acknowledge the uncertainty.
Your goal is to provide reliable, evidence-based answers grounded in the audio content only."""


def format_user_prompt(
    question: str,
    choices: list[str],
    audio_labels: list[str] | None = None,
) -> str:
    labels = audio_labels or ["output", "input1"]
    header = "\n".join(f"<audio {label}>: {{audio_{i + 1}}}" for i, label in enumerate(labels))
    choice_lines = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"{header}\n\n"
        "Based on the objective content of the uploaded audio, answer the following multiple-choice question. "
        "First carefully perceive, analyze, and reason about the audio content, then choose exactly one option "
        "from the list.\n"
        'Return only JSON with keys "reason" and "choice". Put the reason first and the final choice last. '
        "The choice value must be a single uppercase letter identifying the option.\n\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{choice_lines}"
    )


def parse_audio_refs(question: str) -> list[str]:
    return [f"<audio {ref.label}>" for ref in parse_audio_ref_tokens(question)]


def mock_judger_choice(
    right_choice: str,
    choices: list[str],
    *,
    p_correct: float,
    rng: random.Random,
) -> str:
    """Return uppercase letter A..N for a rubric vote."""
    if rng.random() < p_correct:
        idx = choices.index(right_choice)
    else:
        wrong = [i for i, c in enumerate(choices) if c != right_choice]
        idx = rng.choice(wrong) if wrong else 0
    return chr(65 + idx)


def mock_judger_json(choice_letter: str, reason: str = "mock perception") -> str:
    return json.dumps({"reason": reason, "choice": choice_letter})
