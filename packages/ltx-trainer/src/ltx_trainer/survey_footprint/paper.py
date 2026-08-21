"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.survey_footprint.benchmarks import PAPER_TITLE, SUMMARY_V250, benchmarks_bundle
from ltx_trainer.survey_footprint.config import PAPER_ARXIV, TOOL_URL, VERSION


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "version": VERSION,
        "tool_url": TOOL_URL,
        "problem": (
            "Multi-survey science needs fast answers on footprint overlap and whether "
            "catalogue sources fall inside each survey's MOC — without local installs."
        ),
        "method": {
            "footprints": "IVOA MOC / HEALPix (13 built-in surveys, v2.5.0)",
            "views": "Aladin Lite v2 globe + D3 equirectangular (GeoJSON)",
            "engine": "Client-side WASM Rust MOC (union, intersection, filterCoos)",
            "catalogue": "CSV/TSV upload → augmented download with in_<survey> columns",
        },
        "results": {
            "num_surveys": int(SUMMARY_V250["num_surveys"]),
            "union_coverage_deg2": SUMMARY_V250["union_sky_coverage_deg2"],
            "max_area_deg2": SUMMARY_V250["max_survey_area_deg2"],
        },
        "reference_metrics": b,
        "integration": (
            "GOPEX implements MOC-operation stubs, catalogue augmentation, Table 1 anchors, "
            "and Sec. 5 use-case demos. Production UI is the static browser app at tool_url."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.survey_footprint.mock import evaluation_smoke

    return evaluation_smoke()
