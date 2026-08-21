"""Domain subfields (Sec. 4)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class VideoSubfield(str, Enum):
    EGOCENTRIC = "egocentric"
    SPORTS = "sports"
    INSTRUCTIONAL = "instructional"
    MEDICAL = "medical"
    NARRATIVE = "narrative_movies"


SUBFIELD_ANCHORS: dict[VideoSubfield, dict[str, Any]] = {
    VideoSubfield.EGOCENTRIC: {
        "datasets": ["Ego4D", "EgoMask", "DVBench"],
        "methods": ["Ego-R1", "ST-Think", "VideoLLM-EyeWO", "EgoSocial"],
        "stress": "4D dynamics, proactive streaming, ultra-long tool chains",
    },
    VideoSubfield.SPORTS: {
        "datasets": ["SPORTU", "FineQuest", "DeepSport"],
        "methods": ["Unisoccer", "DeepSport agentic think-with-videos"],
        "stress": "brief decisive events, rule-aware reasoning, camera cuts",
    },
    VideoSubfield.INSTRUCTIONAL: {
        "datasets": ["Video-MMMU", "Video-MMLU", "InstructionBench", "DocVideoQA"],
        "methods": ["NoteIt", "InsTALL task graphs"],
        "stress": "procedural progress, slide–speech alignment, knowledge transfer",
    },
    VideoSubfield.MEDICAL: {
        "datasets": ["MM-OR", "SurgViVQA", "EchoCLIP"],
        "methods": ["LLaVA-Surg", "Surgical-LLaVA", "EndoChat", "Sonomate"],
        "stress": "long procedures, rare events, clinically faithful evidence",
    },
    VideoSubfield.NARRATIVE: {
        "datasets": ["MovieQA", "MAD", "MLVU", "SCVBench", "SeriesBench", "MovieCORE"],
        "methods": ["MovieChat", "StoryCoT", "ARC-Chapter"],
        "stress": "plot progression, character tracking, dispersed causal evidence",
    },
}


def subfields_card() -> dict[str, Any]:
    return {
        "subfields": [s.value for s in VideoSubfield],
        "anchors": {s.value: SUBFIELD_ANCHORS[s] for s in VideoSubfield},
    }
