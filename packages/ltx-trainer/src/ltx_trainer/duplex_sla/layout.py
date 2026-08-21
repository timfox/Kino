"""Limitations for DuplexSLA stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Stub holds paper excerpts and layout constants only; no streaming decoder, Step-Audio weights, or DuplexSLA-Bench assets.",
    "Real serving needs causal user frontend, TA4 codec, FIFO action queue, and hardware-specific per-chunk decode budgets (§2.3, Appendix D).",
)
