"""Generative and discriminative MSA prompt templates (arXiv:2606.05713)."""

from __future__ import annotations

GENERATIVE_MSA_TEMPLATE = """You are a multimodal sentiment analyzer.

Given the user's text, video frames, and optional audio from a conversational clip,
estimate the overall sentiment intensity on a continuous scale from -3 (strongly negative)
to +3 (strongly positive).

Respond with a single numeric value only (e.g., 1.25 or -0.50). Do not include units or text."""

GENERATIVE_MSA_TRAINED_SUFFIX = (
    "\n\nUse two decimal places. Valid range: [-3.00, +3.00]."
)

DISCRIMINATIVE_INSTRUCTION = (
    "Multimodal sentiment regression via pooled Thinker hidden state and MLP readout "
    "(no autoregressive decode)."
)

MODALITY_PROMPTS: dict[str, str] = {
    "text_only": "Analyze sentiment from transcript text only.",
    "text_video": "Analyze sentiment from transcript and sampled video frames.",
    "text_video_audio": "Analyze sentiment from transcript, video frames, and audio waveform.",
}


def format_generative_prompt(
    *,
    modality: str = "text_video_audio",
    trained: bool = False,
    utterance_hint: str = "",
) -> str:
    """Build generative decode baseline instruction."""
    prefix = MODALITY_PROMPTS.get(modality, MODALITY_PROMPTS["text_video_audio"])
    body = GENERATIVE_MSA_TEMPLATE
    if utterance_hint:
        body = f"{prefix}\n\nUtterance:\n{utterance_hint.strip()}\n\n{body}"
    else:
        body = f"{prefix}\n\n{body}"
    if trained:
        body += GENERATIVE_MSA_TRAINED_SUFFIX
    return body


def prompt_bundle() -> dict[str, object]:
    return {
        "generative_template": GENERATIVE_MSA_TEMPLATE,
        "modality_configs": list(MODALITY_PROMPTS),
        "discriminative_note": DISCRIMINATIVE_INSTRUCTION,
        "label_range": (-3.0, 3.0),
    }
