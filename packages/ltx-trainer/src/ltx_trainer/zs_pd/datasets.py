"""Corpora — Table 1."""

from __future__ import annotations

from typing import Any


def table_i_datasets() -> list[dict[str, Any]]:
    """Table 1 — four-language PD speech corpora."""
    return [
        {
            "name": "BenSParX",
            "language": "Bengali",
            "pd": 60,
            "hc": 60,
            "task": "Conversation",
            "ref": "[21]",
        },
        {
            "name": "MDVR-KCL",
            "language": "English",
            "pd": 16,
            "hc": 21,
            "task": "Reading text",
            "ref": "[34]",
        },
        {
            "name": "IPVS",
            "language": "Italian",
            "pd": 28,
            "hc": 22,
            "task": "Text dependent utterance",
            "ref": "[35]",
        },
        {
            "name": "NeuroVoz",
            "language": "Spanish",
            "pd": 23,
            "hc": 53,
            "task": "Spontaneous speech",
            "ref": "[36]",
        },
    ]
