"""Downstream integrations (proceduralsky.com HDR skies, CG/VR lighting)."""

from __future__ import annotations

from typing import Any

PROCEDURALSKY_URL = "https://proceduralsky.com"


def proceduralsky_card() -> dict[str, Any]:
    """How CubeDiff panoramas complement proceduralsky-style HDR environment maps."""
    return {
        "partner": PROCEDURALSKY_URL,
        "role": "illumination",
        "cubediff_role": "text- or image-conditioned 360° panorama synthesis (six-face cubemap → ERP)",
        "workflow": [
            "Generate or complete a scene cubemap with CubeDiff (image+text or per-face captions).",
            "Unfold to 2:1 equirectangular (ERP); merge overlaps with SemanticStitch if multi-capture.",
            "Tone-map / grade for artistic HDR skies if needed.",
            "Import as environment map for CG, game engines, or VR (complements catalog HDRIs).",
        ],
        "notes": (
            "Polyhaven and Humus in the CubeDiff training mix overlap proceduralsky use cases; "
            "GOPEX stub does not ship weights or 32-bit EXR export—wire full CubeDiff + ffmpeg "
            "for production HDR pipelines."
        ),
    }
