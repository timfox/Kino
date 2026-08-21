"""Paper tables (arXiv:2605.28733)."""

from __future__ import annotations

from typing import Any

TABLE1_AMAZON_DEMAND: list[dict[str, Any]] = [
    {"term": "Colorfulness", "coef": -2.249},
    {"term": "Colorfulness^2", "coef": 2.118},
    {"term": "Brightness", "coef": -0.281},
    {"term": "Brightness^2", "coef": 0.588},
    {"term": "Symmetry", "coef": -1.074},
    {"term": "Symmetry^2", "coef": 1.664},
    {"term": "Aesthetic Quality", "coef": -2.885},
    {"term": "Aesthetic Quality^2", "coef": 2.951},
]

TABLE2_AMAZON_EDITING: list[dict[str, Any]] = [
    {"model": "Stable Diffusion", "demand": 1.149, "fidelity": 0.224, "uclip": 0.270},
    {"model": "GPT-Image", "demand": 1.007, "fidelity": 0.187, "uclip": 0.322},
    {"model": "Flux", "demand": 1.159, "fidelity": 0.228, "uclip": 0.226},
    {"model": "Utility-Aware Generator", "demand": 1.278, "fidelity": 0.231, "uclip": 0.443},
]

TABLE3_AIRBNB_DEMAND: list[dict[str, Any]] = [
    {"term": "Visual Uniqueness", "coef": 0.288},
    {"term": "Visual Uniqueness^2", "coef": -0.241},
    {"term": "Visual Aesthetics", "coef": 0.978},
    {"term": "Visual Aesthetics^2", "coef": -1.013},
]

TABLE4_AIRBNB_GENERATION: list[dict[str, Any]] = [
    {"model": "Stable Diffusion", "demand": 0.505, "fidelity": 0.213},
    {"model": "DALL·E", "demand": 0.510, "fidelity": 0.224},
    {"model": "Flux", "demand": 0.544, "fidelity": 0.251},
    {"model": "Utility-Aware Generator", "demand": 0.573, "fidelity": 0.268},
]

TABLE5_AIRBNB_EDITING: list[dict[str, Any]] = [
    {"model": "Stable Diffusion", "demand": 0.505, "fidelity": 0.206},
    {"model": "GPT-Image", "demand": 0.508, "fidelity": 0.214},
    {"model": "Flux", "demand": 0.507, "fidelity": 0.209},
    {"model": "Utility-Aware Generator", "demand": 0.521, "fidelity": 0.225},
]

TABLE6_HUMAN_AMAZON: list[dict[str, Any]] = [
    {"model": "Stable Diffusion", "selection_count": 154, "realism": 4.54},
    {"model": "OpenAI", "selection_count": 143, "realism": 4.64},
    {"model": "Flux", "selection_count": 59, "realism": 4.03},
    {"model": "Utility-Aware Generator", "selection_count": 256, "realism": 5.12},
]

TABLE7_REGRESSION_AMAZON: dict[str, float] = {
    "utility_aware_realism": 0.708,
    "utility_aware_professional": 0.741,
    "utility_aware_purchase_logit": 1.038,
}

TABLE8_AIRBNB_RATINGS: list[dict[str, Any]] = [
    {"model": "Flux", "realism": 5.145, "booking": 4.807},
    {"model": "OpenAI", "realism": 4.198, "booking": 4.582},
    {"model": "Stable Diffusion", "realism": 3.268, "booking": 3.423},
    {"model": "Utility-Aware Generator", "realism": 5.263, "booking": 5.039},
]

TABLE9_REGRESSION_AIRBNB: dict[str, float] = {
    "utility_aware_realism": 1.130,
    "utility_aware_uniqueness": 0.503,
    "utility_aware_aesthetic": 0.717,
    "utility_aware_booking": 0.767,
}
