"""Confounders: aging, expression, pose, occlusion, illumination (§5–6)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ChallengeFactor(str, Enum):
    AGING = "aging"
    EXPRESSION = "facial_expression"
    POSE = "pose_variation"
    OCCLUSION = "partial_occlusion"
    ILLUMINATION = "illumination"


INTERNAL_FACTORS = [
    ChallengeFactor.AGING,
    ChallengeFactor.EXPRESSION,
]

EXTERNAL_FACTORS = [
    ChallengeFactor.POSE,
    ChallengeFactor.OCCLUSION,
    ChallengeFactor.ILLUMINATION,
]


def challenge_catalog() -> list[dict[str, Any]]:
    return [
        {
            "factor": ChallengeFactor.AGING.value,
            "type": "internal",
            "summary": "Longitudinal texture and landmark drift; scarce age-progressive training data",
        },
        {
            "factor": ChallengeFactor.EXPRESSION.value,
            "type": "internal",
            "summary": "Non-verbal muscle motion deforms geometry; controlled vs in-the-wild gap",
        },
        {
            "factor": ChallengeFactor.POSE.value,
            "type": "external",
            "summary": "Off-frontal views; pose-specific vs multi-view gallery strategies",
        },
        {
            "factor": ChallengeFactor.OCCLUSION.value,
            "type": "external",
            "summary": "Glasses, beards, hands, veils; part-based and fractal methods",
        },
        {
            "factor": ChallengeFactor.ILLUMINATION.value,
            "type": "external",
            "summary": "Shadows, exposure; gray-level norm, gradient edges, reflectance estimation",
        },
    ]


def mitigation_strategies(factor: ChallengeFactor) -> list[str]:
    mapping: dict[ChallengeFactor, list[str]] = {
        ChallengeFactor.AGING: ["age-invariant subspaces", "cross-age training", "3D shape priors"],
        ChallengeFactor.EXPRESSION: ["expression-neutral synthesis", "muscle-based models", "3DMM"],
        ChallengeFactor.POSE: ["multi-view galleries", "3D model rendering", "frontalization"],
        ChallengeFactor.OCCLUSION: ["part-based features", "occlusion-aware inpainting", "holistic masking"],
        ChallengeFactor.ILLUMINATION: [
            "gray-level normalization",
            "gradient-based edges",
            "reflectance field estimation",
        ],
    }
    return mapping[factor]
