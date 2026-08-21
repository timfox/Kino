"""Fig. 3–4 IoU curve anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dilated_sym_diff.constants import (
    FIG1_DELTA_ALIGN,
    FIG1_IOU_AT_R8,
    FIG1_OPTIMAL_R,
    FIG3_IOU_CURVE,
    FIG4_IOU_CURVE,
    FIG4_WARP,
)


def fig3_iou_curve() -> list[dict[str, int | float]]:
    return [{"r": r, "IoU": iou} for r, iou in FIG3_IOU_CURVE]


def fig4_iou_curve() -> list[dict[str, int | float]]:
    return [{"r": r, "IoU": iou} for r, iou in FIG4_IOU_CURVE]


def summary_anchors() -> dict[str, Any]:
    return {
        "delta_align_fig1": round(FIG1_DELTA_ALIGN, 2),
        "optimal_r_fig1": FIG1_OPTIMAL_R,
        "IoU_at_r8": FIG1_IOU_AT_R8,
        "fig4_warp": dict(FIG4_WARP),
        "fig3_peak_r": 8,
        "fig3_r30_IoU": next(iou for r, iou in FIG3_IOU_CURVE if r == 30),
    }
