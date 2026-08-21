"""Runnable evaluation smoke for Survey Footprint Explorer."""

from __future__ import annotations

from typing import Any

from ltx_trainer.survey_footprint.benchmarks import SUMMARY_V250, benchmarks_bundle
from ltx_trainer.survey_footprint.config import PAPER_ARXIV, VERSION
from ltx_trainer.survey_footprint.surveys import BUILTIN_SURVEYS


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "survey_footprint",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "version": VERSION,
        "benchmarks": benchmarks_bundle(),
        "num_surveys": len(BUILTIN_SURVEYS),
        "union_deg2_ref": SUMMARY_V250["union_sky_coverage_deg2"],
    }
    try:
        from ltx_trainer.survey_footprint.catalogue import augment_catalogue, parse_catalogue_text
        from ltx_trainer.survey_footprint.moc_engine import filter_coordinates, intersection_area_deg2, load_builtin_moc
        from ltx_trainer.survey_footprint.pipeline import evaluation_demo_run

        demo = evaluation_demo_run()
        moc = load_builtin_moc("euclid_dr1")
        inside = filter_coordinates([180.0], [30.0], moc)
        csv = "ra,dec\n12.0,30.0\n"
        headers, rows = parse_catalogue_text(csv)
        _, aug = augment_catalogue(headers, rows, ["euclid_dr1"])
        overlap = intersection_area_deg2(["euclid_dr1", "lsst_wfd", "roman_hlwas"], samples=5000)
        out.update(
            {
                "torch": False,
                "demo_use_case_a_area": demo["use_case_a"]["intersection_area_deg2"],
                "demo_catalogue_sources": demo["use_case_b"]["num_sources"],
                "filter_coos": inside,
                "augmented_columns": len(aug[0]) if aug else 0,
                "triple_overlap_deg2": overlap,
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001 — smoke should report error
        out["error"] = str(exc)
        return out
