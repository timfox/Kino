"""Fixed-dimension CU state features for RL partitioning (Table 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.vtm_partition.qtmtt import ALL_SPLITS, SPLIT_HV, SUB_CU_COUNT


@dataclass
class CUContext:
    """Toy coding-unit context for feature extraction."""

    width: int
    height: int
    qp: int
    ns_rd_cost: float
    ns_rate: float
    ns_distortion: float
    parent_ns_rd_cost: float
    top_rd_cost: float
    left_rd_cost: float
    top_qt_depth: int
    left_qt_depth: int
    split_series_top: tuple[str, ...] = ()
    split_series_left: tuple[str, ...] = ()
    gradient_histogram: tuple[float, ...] | None = None


def _split_series_scalar(series: tuple[str, ...]) -> float:
    """Pack normalized (N_CU, HV) pair from deepest split in the series (Table 2 items 5–6)."""
    if not series:
        return 0.0
    last = series[-1]
    ncu = float(SUB_CU_COUNT.get(last, 1)) / 4.0
    hv = float(SPLIT_HV.get(last, 0))
    return ncu + 0.25 * hv


def extract_state_vector(ctx: CUContext) -> np.ndarray:
    """
    Table 2 — 14-dim state S = [NI(6), PI(3), BI(4), SI(1)].

    HOG (SI) is a single normalized energy stub when full 9-bin HOG is unavailable.
    """
    if ctx.gradient_histogram:
        hog_energy = float(sum(ctx.gradient_histogram)) / max(len(ctx.gradient_histogram), 1)
    else:
        hog_energy = (ctx.width * ctx.height) ** 0.5 / 32.0
    hog_energy = min(hog_energy, 1.0)

    return np.array(
        [
            ctx.top_rd_cost,
            ctx.left_rd_cost,
            float(ctx.top_qt_depth),
            float(ctx.left_qt_depth),
            _split_series_scalar(ctx.split_series_top),
            _split_series_scalar(ctx.split_series_left),
            ctx.parent_ns_rd_cost,
            ctx.ns_rate,
            ctx.ns_distortion,
            float(ctx.width),
            float(ctx.height),
            float(ctx.qp),
            ctx.ns_rd_cost,
            hog_energy,
        ],
        dtype=np.float64,
    )


def feature_groups() -> list[dict[str, Any]]:
    """Human-readable Table 2 layout."""
    return [
        {
            "group": "NI",
            "indices": "1–6",
            "fields": [
                "top RD cost",
                "left RD cost",
                "top QT depth",
                "left QT depth",
                "top SplitSeries (N_CU, HV)",
                "left SplitSeries (N_CU, HV)",
            ],
        },
        {
            "group": "PI",
            "indices": "7–9",
            "fields": ["parent NS RD cost", "NS rate", "NS distortion"],
        },
        {
            "group": "BI",
            "indices": "10–13",
            "fields": ["width", "height", "QP", "current NS RD cost"],
        },
        {"group": "SI", "indices": "14", "fields": ["HOG energy (stub)"]},
    ]
