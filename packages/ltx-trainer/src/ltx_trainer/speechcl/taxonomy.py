"""Representation-centric taxonomy — §2."""

from __future__ import annotations

from enum import Enum
from typing import Any


class GeometryEvolution(str, Enum):
    PRESERVATION = "geometry_preservation"
    EXPANSION = "geometry_expansion"
    ALIGNMENT = "geometry_alignment"
    SPECIALIZATION = "geometry_specialization"


class AdaptationLayer(str, Enum):
    ACOUSTIC_ENCODER = "acoustic_encoder"
    ALIGNMENT = "alignment_layer"
    LANGUAGE_MODEL = "language_model"
    MEMORY = "memory_systems"
    AGENT = "agent_level"


def describe_geometry(kind: GeometryEvolution) -> dict[str, Any]:
    cards: dict[GeometryEvolution, dict[str, Any]] = {
        GeometryEvolution.PRESERVATION: {
            "goal": "Maintain existing latent structure under distributional shift",
            "examples": [
                "new speakers",
                "recording devices",
                "noise profiles",
                "channel effects",
            ],
            "failure_modes": [
                "reduced phonetic separability",
                "collapsed speaker manifolds",
                "weakened paralinguistic structure",
            ],
        },
        GeometryEvolution.EXPANSION: {
            "goal": "Embed unseen information while preserving compatibility",
            "examples": [
                "new languages",
                "accents",
                "vocabularies",
                "speakers",
                "acoustic events",
            ],
            "challenge": "plasticity vs stability in a shared latent space",
        },
        GeometryEvolution.ALIGNMENT: {
            "goal": "Preserve or update cross-space relationships consistently",
            "examples": [
                "speech encoder ↔ frozen LLM",
                "multimodal modules",
                "external memory",
            ],
            "failure_modes": ["alignment drift", "degraded speech-to-text"],
        },
        GeometryEvolution.SPECIALIZATION: {
            "goal": "Reshape shared space for new capabilities on a foundation model",
            "examples": [
                "audio captioning",
                "spoken QA",
                "agentic dialogue",
            ],
            "tension": "capability acquisition vs representational reuse",
        },
    }
    return {"kind": kind.value, **cards[kind]}


def describe_adaptation_layer(layer: AdaptationLayer) -> dict[str, Any]:
    links: dict[AdaptationLayer, tuple[GeometryEvolution, ...]] = {
        AdaptationLayer.ACOUSTIC_ENCODER: (
            GeometryEvolution.PRESERVATION,
            GeometryEvolution.EXPANSION,
        ),
        AdaptationLayer.ALIGNMENT: (GeometryEvolution.ALIGNMENT,),
        AdaptationLayer.LANGUAGE_MODEL: (GeometryEvolution.SPECIALIZATION,),
        AdaptationLayer.MEMORY: (GeometryEvolution.EXPANSION,),
        AdaptationLayer.AGENT: (GeometryEvolution.SPECIALIZATION,),
    }
    notes: dict[AdaptationLayer, str] = {
        AdaptationLayer.ACOUSTIC_ENCODER: "Low-level acoustic geometry: phonetics, speaker, robustness",
        AdaptationLayer.ALIGNMENT: "Cross-modal correspondence speech ↔ text / multimodal",
        AdaptationLayer.LANGUAGE_MODEL: "Higher-level semantic reasoning and instruction following",
        AdaptationLayer.MEMORY: "Incremental knowledge and user-specific information",
        AdaptationLayer.AGENT: "Behavioral policy, planning, tool use",
    }
    return {
        "layer": layer.value,
        "primary_geometry": [g.value for g in links[layer]],
        "note": notes[layer],
    }


def classical_vs_representation_centric() -> dict[str, str]:
    return {
        "classical": "Task / domain / class-incremental (De Lange et al., 2021)",
        "proposed": "How representation geometry evolves under non-stationary acoustics",
    }
