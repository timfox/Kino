"""Handcrafted acoustic features — §2.2."""

from __future__ import annotations

from typing import Any

import numpy as np

# Representative subset of 71 BenSParX-style descriptors (full set in paper)
FEATURE_NAMES = (
    "jitter_local",
    "jitter_rap",
    "shimmer_local",
    "hnr",
    "f0_mean",
    "f0_std",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "speaking_rate",
)


def toy_feature_vector(rng: np.random.Generator, pd: bool) -> dict[str, float]:
    """Synthetic segment features: PD tends to lower HNR / higher jitter."""
    base = {
        "jitter_local": 0.007 if not pd else 0.018,
        "jitter_rap": 0.004 if not pd else 0.012,
        "shimmer_local": 0.04 if not pd else 0.08,
        "hnr": 22.0 if not pd else 14.0,
        "f0_mean": 140.0 + rng.normal(0, 5),
        "f0_std": 25.0 if not pd else 8.0,
        "mfcc_1": -5.2,
        "mfcc_2": 1.1,
        "mfcc_3": -0.3,
        "speaking_rate": 4.5 if not pd else 3.2,
    }
    # Pad to 71 with small noise for stub completeness
    for i in range(len(FEATURE_NAMES), 71):
        base[f"feature_{i}"] = float(rng.normal(0, 1))
    return base


def serialize_features_for_llm(features: dict[str, float]) -> str:
    """Feature–value list for Table 2 LLM prompt {list}."""
    return ", ".join(f"{k}: {features[k]:.6g}" for k in sorted(features.keys())[:20]) + ", ..."


def segment_count(duration_sec: float, segment_sec: float = 10.0) -> int:
    return max(1, int(duration_sec // segment_sec))
