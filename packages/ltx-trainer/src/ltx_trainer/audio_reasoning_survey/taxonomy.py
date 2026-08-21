"""Four-paradigm taxonomy (Fig. 2–3, Sec. I–VII)."""

from __future__ import annotations

from typing import Any


def four_paradigms() -> list[dict[str, Any]]:
    return [
        {
            "id": "audio_to_text",
            "name": "Audio-to-Text reasoning",
            "summary": "Infer textual answers from audio + instruction; acoustic grounding central.",
            "subsections": (
                "Inference-time CoT (e.g., Audio-CoT, SAR-LM)",
                "SFT-based CoT (reasoning-chain datasets)",
                "RL-based CoT (GRPO / verifiable rewards)",
            ),
        },
        {
            "id": "audio_to_speech",
            "name": "Audio-to-Speech reasoning",
            "summary": "Audio-in, speech-out; balance reasoning depth vs latency.",
            "subsections": (
                "Sequential: listen → think → speak (Eq. 5 bottleneck)",
                "Realtime TWL: P(R_t | A_str≤t, X, R_<t) (Eq. 8)",
                "Realtime TWS: P(S_t, R_{t+1} | A, X, S_<t, R_≤t) (Eq. 9)",
            ),
        },
        {
            "id": "audio_visual",
            "name": "Audio-Visual reasoning",
            "summary": "Joint (A, V, X) → Y with optional reasoning R (Eq. 10–12).",
            "subsections": ("SFT omni pipelines", "SFT + distillation", "SFT + RL (e.g., GSPO)"),
        },
        {
            "id": "agentic",
            "name": "Agentic Audio Reasoning",
            "summary": "Planning, tools, memory, reflection; predefined vs dynamic tool-calling.",
            "subsections": (
                "Predefined workflow agents (e.g., Speech-Hands, AudioGenie-Reasoner)",
                "Dynamic tool-calling (e.g., AURA, AudioToolAgent, AuTAgent, Stream RAG)",
            ),
        },
    ]


def agentic_design_patterns() -> tuple[str, ...]:
    """Sec. VII-C design patterns (non-exclusive)."""
    return (
        "Correction",
        "Iterative refinement",
        "Reactive tool-use (Thought–Action–Observation)",
        "Proactive streaming",
        "Data-driven orchestration (RL tool policy)",
        "Asynchronous multi-agent",
    )
