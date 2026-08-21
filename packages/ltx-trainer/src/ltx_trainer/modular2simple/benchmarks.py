"""Paper Table 1 and anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.modular2simple.config import (
    CARLA_VERSION,
    LIBRARY_MODULAR_LOC,
    LIBRARY_REDUCTION_PCT,
    LIBRARY_SCENARIO_COUNT,
    LIBRARY_TRADITIONAL_LOC,
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    SIMPLE_REUSE_LOC,
)
from ltx_trainer.modular2simple.library import TABLE1_LOC, library_summary


PAPER_ANCHORS = {
    "library_scenarios": LIBRARY_SCENARIO_COUNT,
    "modular_loc_total": LIBRARY_MODULAR_LOC,
    "traditional_loc_total": LIBRARY_TRADITIONAL_LOC,
    "code_reduction_pct": LIBRARY_REDUCTION_PCT,
    "simple_behavior_templates": 2,
    "simple_template_loc": list(SIMPLE_REUSE_LOC),
    "carla_version": CARLA_VERSION,
    "cli_cxm": "java Modular2Simple -cxm <inputs> <output.mosc>",
    "cli_cmx": "java Modular2Simple -cmx <input.mosc> <output.xosc>",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "authors": PAPER_AUTHORS,
            "venue": PAPER_VENUE,
            "url": PAPER_URL,
            "repo": PAPER_REPO,
        },
        "table1_loc": TABLE1_LOC,
        "library_summary": library_summary(),
        "anchors": PAPER_ANCHORS,
    }
