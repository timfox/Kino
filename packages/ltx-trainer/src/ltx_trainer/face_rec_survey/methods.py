"""Classical and deep face recognition methods surveyed (§3–4)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class RecognitionDomain(str, Enum):
    D2_STILL = "2d_still"
    D3_SHAPE = "3d_shape"
    VIDEO = "video"


CLASSICAL_METHODS: list[dict[str, Any]] = [
    {"name": "PCA / Eigenfaces", "category": "subspace", "era": "classical"},
    {"name": "ICA", "category": "subspace", "era": "classical"},
    {"name": "LDA", "category": "subspace", "era": "classical"},
    {"name": "Elastic Bunch Graph Matching (EBGM)", "category": "graph", "era": "classical"},
    {"name": "Hidden Markov Models", "category": "sequence", "era": "classical"},
    {"name": "Gabor wavelets", "category": "filter", "era": "classical"},
    {"name": "SVM", "category": "classifier", "era": "classical"},
    {"name": "ANN", "category": "classifier", "era": "classical"},
    {"name": "3D Morphable Model", "category": "3d", "era": "model_based"},
]

DEEP_METHODS: list[dict[str, Any]] = [
    {"name": "CNN", "category": "deep", "era": "deep"},
    {"name": "Guided CNN + metric loss", "category": "deep", "era": "deep"},
    {"name": "Viola-Jones + 3D generic model", "category": "hybrid", "era": "hybrid"},
]

FEATURE_EXTRACTORS: list[str] = [
    "HOG",
    "LBP",
    "LPQ",
    "SIFT",
    "Fourier transforms",
    "Eigenface",
    "Histogram-oriented gradients",
]

DETECTORS: list[str] = [
    "Viola-Jones",
    "HOG-SVM",
    "PCA-based detection",
    "Deep CNN detectors",
]


def table_classical_methods() -> list[dict[str, Any]]:
    return [dict(m) for m in CLASSICAL_METHODS]


def table_deep_methods() -> list[dict[str, Any]]:
    return [dict(m) for m in DEEP_METHODS]


def recognition_pipeline_stages() -> list[dict[str, str]]:
    return [
        {"id": "detection", "role": "Locate faces in image or video frame"},
        {"id": "feature_extraction", "role": "Encode geometry / appearance signature"},
        {"id": "recognition", "role": "Identify (1:N) or verify (1:1) against gallery"},
    ]
