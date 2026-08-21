"""StEM2 cinema SDR→HDR reference tables (arXiv:2604.06276)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stem2_sdr_hdr._bootstrap import ensure_repo_root

ensure_repo_root()

from gopex_datasets.stem2.constants import EXR_CLOSER_RATIO, MEAN_R2, TOTAL_FRAMES
from gopex_datasets.stem2.metrics import (
    table1_luminance_by_scene,
    table2_residual_types,
    table3_color_metrics,
    table4_color_by_luminance_bin,
    table5_decision_map_by_scene,
    table6_exr_correlation,
)

PAPER_ARXIV = "2604.06276"

GLOBAL_STATS: dict[str, float] = {
    "mean_isotonic_r2": MEAN_R2,
    "exr_closer_ratio": EXR_CLOSER_RATIO,
    "total_frames": float(TOTAL_FRAMES),
}


def _table5_with_average_first() -> list[dict[str, Any]]:
    rows = table5_decision_map_by_scene()
    avg = next(r for r in rows if r["scene"] == "full_film_average")
    rest = [r for r in rows if r["scene"] != "full_film_average"]
    return [avg, *rest]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "global_stats": dict(GLOBAL_STATS),
        "table1_luminance_by_scene": table1_luminance_by_scene(),
        "table2_residual_types": table2_residual_types(),
        "table3_color_metrics": table3_color_metrics(),
        "table4_color_by_luminance_bin": table4_color_by_luminance_bin(),
        "table5_decision_map_by_scene": _table5_with_average_first(),
        "table6_exr_correlation": table6_exr_correlation(),
    }
