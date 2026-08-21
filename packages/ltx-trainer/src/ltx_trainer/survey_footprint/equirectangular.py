"""Equirectangular projection helpers (Sec. 2.3.2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.survey_footprint.moc_engine import SphericalPatch
from ltx_trainer.survey_footprint.surveys import SurveyRecord


def patch_to_geojson_polygon(patch: SphericalPatch) -> dict[str, Any]:
    """GeoJSON polygon in plate carrée (RA 360→0, Dec −90→+90)."""
    coords = [
        [patch.ra_max, patch.dec_min],
        [patch.ra_max, patch.dec_max],
        [patch.ra_min, patch.dec_max],
        [patch.ra_min, patch.dec_min],
        [patch.ra_max, patch.dec_min],
    ]
    return {"type": "Polygon", "coordinates": [coords]}


def survey_geojson_features(
    survey: SurveyRecord,
    patches: list[SphericalPatch],
    *,
    color: str = "#4477AA",
) -> list[dict[str, Any]]:
    return [
        {
            "type": "Feature",
            "properties": {"survey_id": survey.survey_id, "label": survey.label, "color": color},
            "geometry": patch_to_geojson_polygon(p),
        }
        for p in patches
    ]
