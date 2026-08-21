"""Ingest open weather API JSON into AirCast-SR-style conditioning summaries."""

from __future__ import annotations

from typing import Any


def ingest_open_meteo_hourly(payload: dict[str, Any]) -> dict[str, Any]:
    """Bridge for procedural sky / AirCast-SR pipelines."""
    try:
        from gopex_datasets.graphcast.inputs import open_meteo_to_conditioning_summary
    except ImportError:
        # Minimal inline fallback when only ltx_trainer on PYTHONPATH
        hourly = payload.get("hourly") or {}
        return {"steps": len(hourly.get("time") or []), "provider": "open-meteo"}
    return open_meteo_to_conditioning_summary(payload)


def sky_driver_fields(summary: dict[str, Any]) -> dict[str, list[float]]:
    """Extract lists suited to cloud-map / timelapse drivers."""
    steps = summary.get("steps") or []
    return {
        "cloud_cover_pct": [float(s["cloud_cover_pct"] or 0) for s in steps],
        "precip_mm": [float(s["precip_mm"] or 0) for s in steps],
        "wind_speed_proxy": [
            float((s.get("u10_ms") or 0) ** 2 + (s.get("v10_ms") or 0) ** 2) ** 0.5 for s in steps
        ],
    }
