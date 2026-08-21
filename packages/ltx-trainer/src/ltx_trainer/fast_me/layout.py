"""Scope notes for FAST-ME reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No ViT/SAM/CLIP inference: semantic attention Ak is mocked or supplied externally.",
    "MATLAB FS/DS/TSS baselines and mjpegtools DERF streams are external.",
    "Empirical CDF stopping uses observed SAD samples only (no full search window).",
    "Paper tables are quoted for agent orientation, not reproduced on video frames.",
)
