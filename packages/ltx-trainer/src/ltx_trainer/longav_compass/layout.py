"""Scope notes for LongAV-Compass reference implementation."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No Gemini 3.1 Pro API calls: MLLM judges are deterministic stubs for agents and smoke tests.",
    "DINO-v2, ArcFace, CLIP, and ImageBind are represented by cosine-similarity proxies, not loaded checkpoints.",
    "284-case annotation JSON and model output videos are not bundled; use upstream GitHub release when available.",
    "Leaderboard balanced scores use a simplified composite; paper uses full diagnostic protocol per task.",
)
