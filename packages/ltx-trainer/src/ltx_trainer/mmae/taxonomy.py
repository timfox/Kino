"""MMAE taxonomy distributions (Figure 2) and dimension helpers."""

from __future__ import annotations

from typing import Any

MODALITY_DISTRIBUTION: dict[str, float] = {
    "sound": 21.2,
    "music": 21.3,
    "speech": 21.3,
    "mix": 36.2,
    "sound-speech": 9.8,
    "sound-music": 8.9,
    "music-speech": 8.8,
    "sound-music-speech": 8.8,
}

COMPLEXITY_DISTRIBUTION: dict[str, float] = {
    "single": 50.1,
    "multiple": 49.9,
    "multi-instruction": 11.3,
    "multi-part": 10.0,
    "multi-hop": 10.0,
    "multi-round": 9.7,
    "multi-audio": 8.9,
}

OPERATION_DISTRIBUTION: dict[str, float] = {
    "local": 50.5,
    "global": 30.4,
    "mix": 19.1,
    "replacement": 12.5,
    "extraction": 11.8,
    "alteration": 9.6,
    "removal": 8.6,
    "addition": 8.0,
    "foreground change": 12.7,
    "background change": 8.1,
}

RUBRIC_PRINCIPLES: tuple[str, ...] = (
    "Completeness",
    "Atomicity",
    "Orthogonality",
    "Objectivity",
)


def taxonomy_summary() -> dict[str, Any]:
    return {
        "modalities": list(MODALITY_DISTRIBUTION.keys()),
        "complexity_levels": [k for k in COMPLEXITY_DISTRIBUTION if k != "multiple"],
        "granularity": ["local", "global"],
        "local_operations": ["addition", "removal", "replacement", "extraction", "alteration"],
        "global_operations": ["background change", "foreground change", "alteration"],
        "modality_distribution_pct": dict(MODALITY_DISTRIBUTION),
        "complexity_distribution_pct": dict(COMPLEXITY_DISTRIBUTION),
        "operation_distribution_pct": dict(OPERATION_DISTRIBUTION),
        "rubric_principles": list(RUBRIC_PRINCIPLES),
    }
