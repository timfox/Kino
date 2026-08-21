"""Illustrative workflows (Sec. 5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.survey_footprint.app_state import AppState
from ltx_trainer.survey_footprint.catalogue import augment_catalogue, parse_catalogue_text
from ltx_trainer.survey_footprint.config import USE_CASE_SURVEYS, SurveyFootprintConfig
from ltx_trainer.survey_footprint.equirectangular import survey_geojson_features
from ltx_trainer.survey_footprint.moc_engine import (
    SphericalPatch,
    intersection_area_deg2,
    load_builtin_moc,
    load_custom_moc,
    union_area_deg2,
)
from ltx_trainer.survey_footprint.surveys import SURVEY_BY_ID
from ltx_trainer.survey_footprint.themes import palette_for_theme


def use_case_a_overlap(cfg: SurveyFootprintConfig | None = None) -> dict[str, Any]:
    """Multi-survey footprint comparison (Sec. 5.1)."""
    cfg = cfg or SurveyFootprintConfig()
    ids = list(USE_CASE_SURVEYS)
    area = intersection_area_deg2(ids, samples=cfg.mc_samples)
    state = AppState(
        selected_surveys=ids,
        survey_order=ids,
        cross_match_only=True,
    )
    colors = palette_for_theme("rainbow", len(ids))
    geo_layers = []
    for sid, color in zip(ids, colors, strict=True):
        moc = load_builtin_moc(sid)
        rec = SURVEY_BY_ID[sid]
        geo_layers.extend(survey_geojson_features(rec, moc.patches, color=color))
    return {
        "surveys": ids,
        "intersection_area_deg2": area,
        "cross_match_only": True,
        "geojson_feature_count": len(geo_layers),
        "app_state": state.to_local_storage(),
    }


def use_case_b_catalogue_augment(
    csv_text: str,
    survey_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Catalogue membership columns (Sec. 5.2)."""
    ids = survey_ids or list(USE_CASE_SURVEYS)
    headers, rows = parse_catalogue_text(csv_text)
    new_headers, aug = augment_catalogue(headers, rows, ids)
    n_in_all = sum(1 for r in aug if all(r.get(f"in_{s}") for s in ids))
    return {
        "num_sources": len(aug),
        "columns_added": [f"in_{s}" for s in ids],
        "in_all_three": n_in_all,
        "headers": new_headers,
        "sample_row": aug[0] if aug else {},
    }


def use_case_c_custom_moc(
    custom_patches: list[SphericalPatch],
    survey_ids: list[str],
    *,
    cfg: SurveyFootprintConfig | None = None,
) -> dict[str, Any]:
    """Custom MOC upload + overlap (Sec. 5.3)."""
    cfg = cfg or SurveyFootprintConfig()
    custom = load_custom_moc("custom_upload", custom_patches)
    builtin = [load_builtin_moc(s) for s in survey_ids]
    combined_ids = ["custom_upload", *survey_ids]
    # intersection: custom patch AND each builtin
    area = intersection_area_deg2(survey_ids, samples=cfg.mc_samples)
    custom_hits = sum(
        1
        for _ in range(1000)
        if custom.contains(180.0, 30.0)  # smoke point
    )
    return {
        "custom_moc_id": custom.survey_id,
        "selected_builtin": survey_ids,
        "builtin_intersection_deg2": area,
        "custom_patch_count": len(custom_patches),
        "combined_ids": combined_ids,
        "builtin_mocs_loaded": len(builtin),
    }


def evaluation_demo_run(cfg: SurveyFootprintConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SurveyFootprintConfig()
    csv = "RA_deg,DEC_deg\n180.0,30.0\n150.0,10.0\n200.0,-20.0\n"
    a = use_case_a_overlap(cfg)
    b = use_case_b_catalogue_augment(csv)
    c = use_case_c_custom_moc(
        [SphericalPatch(170, 190, 25, 35)],
        ["euclid_dr1", "lsst_wfd"],
        cfg=cfg,
    )
    all_ids = [s.survey_id for s in SURVEY_BY_ID.values()]
    return {
        "use_case_a": a,
        "use_case_b": b,
        "use_case_c": c,
        "union_13_surveys_deg2": union_area_deg2(all_ids, samples=min(cfg.mc_samples, 5000)),
    }
