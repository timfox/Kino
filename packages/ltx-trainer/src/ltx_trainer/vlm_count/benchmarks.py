"""Paper excerpt benchmarks — visual counting bottleneck (arXiv:2605.30170)."""

from __future__ import annotations

from typing import Any

# Synthetic toy VLM — visual vs text accuracy by regime (Fig. 2 qualitative)
SYNTHETIC_REGIME_ACCURACY = {
    "text": {"ID": 100.0, "VE": 100.0, "FE": 0.0},
    "vision": {"ID": 100.0, "VE": 0.0, "FE": 0.0},
}

# Comparative counting — vision accuracy stays high in VE (Fig. 4)
SYNTHETIC_COMPARE_VISION = {"ID": 98.0, "VE": 92.0, "FE": 55.0}

# Qwen3-VL 6×6 — Fig. 7 approximate endpoints
QWEN3VL_6X6 = {
    "text_min_accuracy_pct": 50.0,
    "vision_at_n8_pct": 5.0,
    "vision_at_n15_pct": 0.0,
}

# Mechanism — Fig. 3 / Fig. 8
MECHANISM = {
    "synthetic_vision_gap_mean": 0.0,
    "synthetic_language_gap_ve_onset": 50,
    "qwen_vision_gap_mean": 0.1,
    "circuit_head_disjoint_pct": 95.7,
    "steering_accuracy_k1_4_pct": 100.0,
    "steering_accuracy_k5_pct": 77.2,
    "layer_probe_accuracy_pct": 100.0,
}

# Error attractors (Fig. 5) — top predicted tokens in VE
VE_ERROR_ATTRACTORS = [49, 99, 90, 9, 95, 96, 97, 98]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "synthetic_regime_accuracy": SYNTHETIC_REGIME_ACCURACY,
        "synthetic_compare_vision": SYNTHETIC_COMPARE_VISION,
        "qwen3vl_6x6": QWEN3VL_6X6,
        "mechanism": MECHANISM,
        "ve_error_attractors": VE_ERROR_ATTRACTORS,
        "fractured_magnitude_hypothesis": True,
        "upstream_code": "https://github.com/Russellpang/semproj",
    }
