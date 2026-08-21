"""Scope notes for Bernini reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No MLLM/DiT training or inference: planner mask schedule, SA-3D RoPE toy, and guidance combine are math stubs only.",
    "Bernini-Bench human/MLLM evaluation and full data pipelines (Sec. 3) are not runnable in this package.",
    "BT leaderboard and benchmark tables quote paper figures for agent-facing smoke, not reproduced from video runs.",
)
