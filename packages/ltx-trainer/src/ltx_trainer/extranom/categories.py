"""ExtrAnom anomaly categories and per-class statistics (Table 2, Fig. 1)."""

from __future__ import annotations

from typing import Any


def anomaly_category_counts() -> dict[str, int]:
    """Video counts per label (501 anomalous + 500 normal)."""
    return {
        "normal": 500,
        "harassment": 189,
        "chain_snatching": 176,
        "kidnapping": 74,
        "stalking": 39,
        "assassination": 23,
    }


def table_category_statistics() -> dict[str, dict[str, float | int]]:
    """Table 2 — frames and capture-condition percentages per category."""
    return {
        "assassination": {
            "videos": 23,
            "frames": 18422,
            "low_light_pct": 8.7,
            "long_shot_pct": 8.7,
            "low_resolution_pct": 17.4,
            "well_lit_pct": 65.2,
        },
        "kidnapping": {
            "videos": 74,
            "frames": 60785,
            "low_light_pct": 10.8,
            "long_shot_pct": 16.2,
            "low_resolution_pct": 8.1,
            "well_lit_pct": 64.9,
        },
        "stalking": {
            "videos": 39,
            "frames": 21142,
            "low_light_pct": 12.8,
            "long_shot_pct": 17.9,
            "low_resolution_pct": 15.5,
            "well_lit_pct": 53.8,
        },
        "harassment": {
            "videos": 189,
            "frames": 104477,
            "low_light_pct": 9.5,
            "long_shot_pct": 14.8,
            "low_resolution_pct": 9.5,
            "well_lit_pct": 66.2,
        },
        "chain_snatching": {
            "videos": 176,
            "frames": 31545,
            "low_light_pct": 4.0,
            "long_shot_pct": 15.3,
            "low_resolution_pct": 17.6,
            "well_lit_pct": 63.1,
        },
    }


def category_share_of_anomalous() -> dict[str, float]:
    """Share of 501 anomalous videos."""
    total = 501
    counts = {k: v for k, v in anomaly_category_counts().items() if k != "normal"}
    return {k: 100.0 * v / total for k, v in counts.items()}


def category_share_of_total() -> dict[str, float]:
    """Share of all 1001 videos (paper abstract, Sec. 1)."""
    total = 1001
    counts = anomaly_category_counts()
    return {k: 100.0 * v / total for k, v in counts.items()}
