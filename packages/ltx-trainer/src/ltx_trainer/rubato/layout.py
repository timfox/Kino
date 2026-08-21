"""Scope notes for Rubato / InterMo reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No Fast Conformer / Transformer weights: this package encodes InterMo conventions, dialects, and paper metrics only.",
    "Full OMR-NED evaluation requires engraved primitives pipeline (Sheet Music Benchmark); not bundled.",
    "Audio synthesis (DawDreamer, VirtuosoNet, VST augmentation) and MAESTRO/(n)ASAP/PDMX ingestion are external.",
    "Interval-piece SentencePiece vocabulary is not shipped; use paper vocabulary sizes for planning.",
)
