"""Limitations for PlanRAG-Audio stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Stub omits real OWSM, pyannote, BEATs, Odyssey pipelines and hybrid SQL generator; numbers are paper excerpts.",
    "Keyword retrieval and constrained planning schema are described in the paper but not executable against a live DB here.",
)
