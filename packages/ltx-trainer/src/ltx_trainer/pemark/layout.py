"""Scope notes for PEMark reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No real HTTP proxying or OpenResty integration; only pure-Python permutation watermarking.",
    "No JSON parser/serializer canonicalization beyond Python dict ordering; treat inputs as key lists or dicts.",
    "Robustness curves are paper excerpts; this stub provides toy simulators for deletion/tamper/insert to exercise extraction.",
)

