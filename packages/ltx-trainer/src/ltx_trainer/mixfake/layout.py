"""Scope notes for MixFake reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No XLSR-AASIST training or MixFake audio I/O: signal math and paper tables only.",
    "HHT uses scalar IF stubs rather than full EMD + Hilbert on prompt embeddings.",
    "EER values are quoted from Tables II–V, not measured on local audio.",
    "Dataset download and mixing pipeline live in the upstream MixFake repository.",
)
