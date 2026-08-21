"""Scope notes for LMDM reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No Stable Audio Open / DiT finetuning: routing, ARC-Forcing math, and Table 1 only.",
    "KV-cache inference and ONNX/JUCE deployment are external to this package.",
    "FD/KL/CLAP metrics are quoted from the paper, not reproduced locally.",
    "Stable Audio Open finetuning and ARC-Forcing training loops are external.",
)
