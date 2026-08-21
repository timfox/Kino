"""Paper limitations and scope notes for DIVA reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Evaluation in paper focuses on 1.5B–8B single-backbone UMMs; scaling to larger models is unverified.",
    "Post-training requires paired image-text anchors and two complementary flows (captioning + masked inpainting).",
    "Middle-layer range (default layers 8–18) is architecture-specific; wrong ranges add cost without gains.",
    "Reference implementation here does not train Nexus-Gen, Show-o, or Liquid — only reproduces objectives and tables.",
    "Video and interleaved multimodal generation are out of scope for the published work.",
)
