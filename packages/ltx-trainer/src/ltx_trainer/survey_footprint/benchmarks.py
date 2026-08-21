"""Reference metrics from Ahad et al. (arXiv:2605.11099)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.survey_footprint.config import (
    FULL_SKY_DEG2,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    TOOL_URL,
    VERSION,
)
from ltx_trainer.survey_footprint.surveys import BUILTIN_SURVEYS

# Table 1 — built-in survey areas (deg²)
TABLE1_SURVEYS: list[dict[str, Any]] = [
    {
        "survey_id": s.survey_id,
        "label": s.label,
        "wavelength": s.wavelength,
        "area_deg2": s.area_deg2,
        "reference": s.reference,
    }
    for s in BUILTIN_SURVEYS
]

# Sec. 6.1 summary statistics (v2.5.0 MOC set)
SUMMARY_V250: dict[str, float] = {
    "sum_individual_areas_deg2": 80_880.1,
    "union_sky_coverage_deg2": 34_104.7,
    "min_survey_area_deg2": 7.7,
    "max_survey_area_deg2": 21_524.4,
    "num_surveys": 13,
}

STACK = {
    "frontend": ["vanilla JS ES2022", "Aladin Lite v2", "D3.js v7", "jsPDF 2.5", "jQuery 3.7"],
    "moc_engine": "WebAssembly Rust cds-moc-rust (client-side)",
    "hosting": "static SPA — no server computation",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "version": VERSION,
            "tool_url": TOOL_URL,
        },
        "full_sky_deg2": FULL_SKY_DEG2,
        "table1_surveys": TABLE1_SURVEYS,
        "summary_v250": SUMMARY_V250,
        "stack": STACK,
    }
