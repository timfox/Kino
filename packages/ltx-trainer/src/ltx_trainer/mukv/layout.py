"""Scope notes for MuKV reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No real video decoding, MLLM prefill, KV-cache storage, or online QA; only toy math stubs for DCP and retrieval.",
    "Tables/Figures are paper excerpts used for agent-facing smoke tests, not reproduced from runs in this repo.",
    "FFT frequency scores are implemented as tiny toy DFT for small sequences (no torch/numpy dependency).",
)

