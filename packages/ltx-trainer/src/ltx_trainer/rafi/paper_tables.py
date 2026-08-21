"""Paper Fig. 8 / bandwidth excerpts (arXiv:2605.30294)."""

from __future__ import annotations

from typing import Any


def fig8_bandwidth_utilization() -> list[dict[str, Any]]:
    """Alps GH200: sustained vs OSU peak (%), 44-byte rays."""
    return [
        {"regime": "intra-node", "gpus": 4, "link": "NVLink", "peak_gbps": 133.0, "sustained_gbps": 100.0, "pct_peak": 75.0},
        {"regime": "inter-node", "gpus": 16, "link": "Slingshot", "peak_gbps": 24.0, "sustained_gbps": 20.0, "pct_peak": 80.0},
    ]


def ray_throughput() -> dict[str, float]:
    return {
        "ray_bytes": 44.0,
        "intranode_rays_per_sec": 2.1e9,
        "internode_rays_per_sec": 5.0e8,
    }
